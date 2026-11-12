"""File-only diagnostics and fixed-model SPSA comparison; no hardware modes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware_capture.optimization.diagnostics import saved_diagnostics
from hardware_capture.optimization.simulation import compare_saved_run, write_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    diagnostic = subparsers.add_parser("diagnostics")
    diagnostic.add_argument("run", type=Path)
    diagnostic.add_argument("--output", type=Path)
    compare = subparsers.add_parser("compare")
    compare.add_argument("run", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--updates", type=int, default=1000)
    compare.add_argument("--samples", type=int, default=600)
    compare.add_argument("--seeds", type=int, nargs="+", default=[20260820, 20260821, 20260822])
    compare.add_argument("--workers", type=int, default=3, help="Independent numerical seed processes")
    compare.add_argument("--comparison", choices=("learning-rates", "step-caps", "amplitude-caps"), default="learning-rates",
                         help="Change learning rates or remove both update caps, holding other settings fixed")
    compare.add_argument("--learning-rate-factors", type=float, nargs="+",
                         help="Multipliers for both saved learning rates (default: 1 0.5)")
    compare.add_argument("--amplitude-step-caps", type=float, nargs="+",
                         help="Amplitude-logit caps to compare at one fixed learning-rate factor")
    args = parser.parse_args(argv)
    if args.mode == "compare" and (args.workers < 1 or len(set(args.seeds)) != len(args.seeds)):
        parser.error("Workers must be positive and seeds unique")
    if args.mode == "diagnostics":
        result = saved_diagnostics(args.run, args.output)
    elif args.workers == 1 or len(args.seeds) == 1:
        result = compare_saved_run(args.run, args.output, updates=args.updates, samples=args.samples,
                                   seeds=args.seeds, comparison=args.comparison,
                                   learning_rate_factors=args.learning_rate_factors,
                                   amplitude_step_caps=args.amplitude_step_caps)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        folders = [args.output / f"seed_group_{seed}" for seed in args.seeds]
        with ProcessPoolExecutor(max_workers=min(args.workers, len(args.seeds))) as pool:
            futures = [pool.submit(compare_saved_run, args.run, folder, updates=args.updates,
                                   samples=args.samples, seeds=(seed,), comparison=args.comparison,
                                   learning_rate_factors=args.learning_rate_factors,
                                   amplitude_step_caps=args.amplitude_step_caps)
                       for seed, folder in zip(args.seeds, folders)]
            result = [row for future in futures for row in future.result()]
        histories = {}
        for folder in folders:
            for history_path in folder.glob("seed_*/history.csv"):
                with history_path.open(newline="", encoding="utf-8") as stream:
                    histories[history_path.parent.name] = list(csv.DictReader(stream))
        write_comparison(args.output, histories, result)
        manifest = json.loads((folders[0] / "manifest.json").read_text(encoding="utf-8"))
        manifest.update(seeds=args.seeds, workers=min(args.workers, len(args.seeds)),
                        seed_directories=[folder.name for folder in folders])
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
