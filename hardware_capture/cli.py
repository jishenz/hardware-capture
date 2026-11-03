"""Single CLI for all project workflows."""
import argparse
import json

from hardware_capture.api import PROJECTS, plan, run
from hardware_capture.workflows.common import json_ready


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hc", description="Configure once; generate, analyze, calibrate, edit and optimize RF pulses.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List available workflows")
    for name in ["check", "run"]:
        command = sub.add_parser(name)
        command.add_argument("project", choices=PROJECTS)
        command.add_argument("--config", "-c", required=True)
        if name == "run":
            command.add_argument("--live", action="store_true", help="Permit the configured hardware workflow to connect and send commands")
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            result = PROJECTS
        elif args.command == "check":
            result = plan(args.project, args.config)
        else:
            result = run(args.project, args.config, live=args.live)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as error:
        parser.exit(2, f"Configuration/workflow error: {error}\n")
    print(json.dumps(json_ready(result), indent=2, allow_nan=False))
    return 0
