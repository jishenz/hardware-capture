"""Capture a time-windowed waveform from a RIGOL DHO scope and save CSV.

Example:
    uv run python hardware_capture/oscilloscope/examples/capture_time_window.py --t1 -1us --t2 3us
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware_capture.oscilloscope.rigol_dho import (
    DEFAULT_ARMED_TIMEOUT_S,
    DEFAULT_CHUNK_POINTS,
    DEFAULT_IP,
    DEFAULT_TIMEOUT_MS,
    DEFAULT_TRIGGER_POLL_S,
    DEFAULT_TRIGGER_TIMEOUT_S,
    RigolDHO,
    ScopeTimeoutError,
    ScopeWaveform,
    WaveformCaptureConfig,
    parse_time_seconds,
    save_waveform_csv,
    timestamped_capture_path,
)

WINDOW_START = "-23.8us"
WINDOW_END = "100us"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save a selected RIGOL DHO waveform time window as CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ip", default=DEFAULT_IP)
    parser.add_argument("--resource")
    parser.add_argument("--socket", action="store_true")
    parser.add_argument("--channel", default="1")
    parser.add_argument("--t1", default=WINDOW_START)
    parser.add_argument("--t2", default=WINDOW_END)
    parser.add_argument("--output", type=Path, default=timestamped_capture_path())
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--chunk-points", type=int, default=DEFAULT_CHUNK_POINTS)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--shot-command")
    parser.add_argument("--read-current", action="store_true")
    parser.add_argument("--allow-clamp", action="store_true")
    parser.add_argument("--no-arm-confirm", action="store_true")
    parser.add_argument("--no-rearm-after-capture", action="store_true")
    parser.add_argument(
        "--trigger-timeout-s", type=float, default=DEFAULT_TRIGGER_TIMEOUT_S
    )
    parser.add_argument(
        "--armed-timeout-s", type=float, default=DEFAULT_ARMED_TIMEOUT_S
    )
    parser.add_argument("--poll-s", type=float, default=DEFAULT_TRIGGER_POLL_S)
    return parser


def _time_window(args: argparse.Namespace) -> tuple[float, float]:
    t1 = parse_time_seconds(args.t1)
    t2 = parse_time_seconds(args.t2)
    if t1 is None or t2 is None or t2 <= t1:
        raise SystemExit("--t2 must be greater than --t1.")
    return t1, t2


def _shot_path(path: Path, shot_index: int, shots: int) -> Path:
    if shots == 1:
        return path
    return path.with_name(f"{path.stem}_shot{shot_index + 1:04d}{path.suffix}")


def _start_shot(args: argparse.Namespace, shot_index: int) -> None:
    print(f"Scope armed for shot {shot_index + 1}/{args.shots}.")
    if args.shot_command:
        subprocess.run(args.shot_command, shell=True, check=True)
    else:
        print("Waiting for external trigger or TNMR shot...")


def _capture(
    scope: RigolDHO,
    config: WaveformCaptureConfig,
    args: argparse.Namespace,
    shot_index: int,
) -> tuple[ScopeWaveform, str]:
    if args.read_current:
        return scope.capture_waveform(config), "read-current"

    waveform = scope.capture_single_waveform(
        config,
        start_shot=lambda: _start_shot(args, shot_index),
        trigger_timeout_s=args.trigger_timeout_s,
        armed_timeout_s=args.armed_timeout_s,
        poll_s=args.poll_s,
        confirm_armed=not args.no_arm_confirm,
        rearm_after_read=not args.no_rearm_after_capture,
    )
    return waveform, scope.trigger_status()


def main() -> int:
    args = build_parser().parse_args()
    if args.shots < 1:
        raise SystemExit("--shots must be at least 1.")

    t1, t2 = _time_window(args)
    config = WaveformCaptureConfig(
        channel=args.channel,
        chunk_points=args.chunk_points,
        window_start_s=t1,
        window_end_s=t2,
        allow_clamp=args.allow_clamp,
    )
    with RigolDHO(
        resource=args.resource,
        ip=args.ip,
        use_socket=args.socket,
        timeout_ms=args.timeout_ms,
    ) as scope:
        for shot_index in range(args.shots):
            try:
                waveform, status = _capture(scope, config, args, shot_index)
            except ScopeTimeoutError as exc:
                print(f"Shot {shot_index + 1} failed: {exc}")
                continue

            saved_path = save_waveform_csv(
                _shot_path(args.output, shot_index, args.shots),
                waveform,
                metadata={
                    "shot_index": shot_index + 1,
                    "shots_requested": args.shots,
                    "scope": waveform.source,
                    "channel": waveform.channel,
                    "start_point": waveform.start_point,
                    "stop_point": waveform.stop_point,
                    "trigger_status_after_read": status,
                },
            )
            print(
                f"Saved shot {shot_index + 1}/{args.shots}: "
                f"{len(waveform.voltage)} samples to {saved_path}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
