"""Prepare or run the BB1-seeded composite optimizer (4000 updates by default).

The legacy filename is retained so existing operator commands still work.

Default: file-only configuration preview. --live is for the HUMAN OPERATOR
after confirming the exact devices/channel, approved safe state, laboratory
setup, table playback timing, and cleanup prerequisites. Agents must never
execute the hardware branch. Approval never authorizes agent hardware control.

--plot-only RUN_DIRECTORY regenerates a saved loss curve without device access.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware_capture.api import plan, run
from hardware_capture.configuration import Experiment
from hardware_capture.optimization.history import plot_saved_history

DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def load_config(path: Path = DEFAULT_CONFIG) -> Experiment:
    """Keep configured update count and connections, with a BB1 initial seed."""
    return Experiment.load(path).with_updates(
        pulse={"kind": "bb1", "angle_deg": 180, "axis_phase_deg": 0},
        composite={"seed": "configured"},
    )


def preview(config: Experiment) -> dict:
    description = plan("optimize-composite", config)
    scope, tnmr = config.section("scope"), config.section("tnmr")
    return {
        "updates": config.require("optimization", "updates"),
        "initial_pulse": "symmetric BB1 X(pi)",
        "duration_s": config.require("pulse", "duration_s"),
        "optimizer": description["optimizer"],
        "repeated_pairs_per_record": description["composite_schedule"]["repetitions"],
        "capture_window": description["capture_window"],
        "capture_reference_start_s": config.section("composite").get(
            "capture_reference_start_s"
        ),
        "scope_resource": scope["resource"],
        "scope_channel": scope.get("channel", "CHAN1"),
        "tnmr_document": tnmr["path"],
        "target_tables": [tnmr["amplitude_table"], tnmr["phase_table"]],
        "output_parent": config.section("output").get(
            "directory", str(ROOT / "outputs")
        ),
        "loss_outputs": ["history.csv", "loss_curve.png"],
        "capture_retention": config.section("composite").get("capture_retention", "selected"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--live",
        action="store_true",
        help="Human operator only: acquire and optimize after setup approval",
    )
    mode.add_argument(
        "--plot-only",
        type=Path,
        metavar="RUN_DIRECTORY",
        help="Offline: regenerate loss_curve.png from saved history.csv",
    )
    args = parser.parse_args(argv)
    if args.plot_only is not None:
        print(plot_saved_history(args.plot_only))
        return 0
    config = load_config(args.config)
    print(json.dumps(preview(config), indent=2), flush=True)
    if not args.live:
        print(
            "Preview only. The human operator may use --live after approving the physical setup."
        )
        return 0
    result = run("optimize-composite", config, live=True)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
