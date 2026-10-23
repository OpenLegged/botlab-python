"""Generic control command line. Robot-specific actions come from capabilities."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .client import ControlClient
from .protocol import FAILED, ControlError, parse_json


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ControlError("invalid_arguments", message)


def _common(parser: argparse.ArgumentParser, *, subcommand: bool = False) -> None:
    default = argparse.SUPPRESS if subcommand else None
    parser.add_argument("--project", default=argparse.SUPPRESS if subcommand else ".", help="project directory (default: cwd)")
    parser.add_argument("--endpoint", default=default, help="explicit loopback middleware HTTP origin")
    parser.add_argument("--instance", default=default, help="bind a middleware instance")
    parser.add_argument("--session", default=default, help="bind a runtime session")
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS if subcommand else False,
                        help="write only structured JSON to stdout")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="botlab control", description="Use any project middleware implementing Control API v1.")
    _common(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("instances", "capabilities", "sessions", "state", "connection", "call", "move", "action", "wait", "retry", "stop"):
        sub = subparsers.add_parser(name)
        _common(sub, subcommand=True)
        if name in {"call", "move", "action", "stop"}:
            sub.add_argument("--request-id", help="stable ID for explicit retries")
        if name in {"call", "action"}:
            sub.add_argument("name", help="declared capability ID or action alias")
        if name == "call":
            args = sub.add_mutually_exclusive_group()
            args.add_argument("--args", default="{}", help="capability arguments as JSON")
            args.add_argument("--args-file", type=Path, help="read capability JSON from a file")
        elif name == "move":
            sub.add_argument("--vx", type=float, required=True)
            sub.add_argument("--vy", type=float, default=0)
            sub.add_argument("--yaw-rate", type=float, default=0)
            sub.add_argument("--duration", type=float, required=True)
        elif name in {"wait", "retry"}:
            sub.add_argument("--id", required=True)
            if name == "wait":
                sub.add_argument("--timeout", type=float, default=1, help="bounded wait in seconds (0..30)")
        elif name == "connection":
            sub.add_argument("--show-token", action="store_true", help="explicitly reveal the temporary pairing token")
    return parser


def run(args: argparse.Namespace) -> dict:
    client = ControlClient(args.project, endpoint=args.endpoint, instance=args.instance, session=args.session)
    if args.command in {"instances", "capabilities", "sessions", "state"}:
        return getattr(client, args.command)()
    if args.command == "connection":
        return client.connection(args.show_token)
    if args.command == "retry":
        return client.retry(args.id)
    if args.command == "wait":
        if not math.isfinite(args.timeout) or not 0 <= args.timeout <= 30:
            raise ControlError("invalid_arguments", "--timeout must be finite and between 0 and 30 seconds.")
        return client.wait(args.id, args.timeout)
    if args.command == "call":
        values = parse_json(args.args_file.read_text() if args.args_file else args.args)
        return client.call(args.name, values, request_id=args.request_id)
    if args.command == "action":
        return client.call(args.name, {}, request_id=args.request_id, alias=True)
    if args.command == "move":
        values = {"vx": args.vx, "vy": args.vy, "yaw_rate": args.yaw_rate, "duration_s": args.duration}
        return client.call("motion.twist", values, request_id=args.request_id)
    return client.call("control.stop", {}, request_id=args.request_id)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    json_output = "--json" in argv
    try:
        args = build_parser().parse_args(argv)
        result = run(args)
        code = 3 if result.get("command", {}).get("status") in FAILED else 0
    except ControlError as error:
        result, code = error.payload(), 2
    except (OSError, UnicodeError) as error:
        result, code = ControlError("local_io_error", "Cannot access the requested local control data.").payload(), 2
        if not json_output:
            print(f"botlab control: {error.__class__.__name__}", file=sys.stderr)
    if not json_output and not result.get("ok"):
        print(f"botlab control: {result['error']['message']}", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=None if json_output else 2))
    return code
