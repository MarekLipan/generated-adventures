"""Command-line entry point: `adventures` starts the game's web UI."""

import argparse
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="adventures", description="Run the Generated Adventures web UI."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind; 0.0.0.0 makes it reachable from your LAN",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload", action="store_true", help="restart on code changes (development)"
    )
    args = parser.parse_args()

    import uvicorn  # deferred so `--help` doesn't pay for the import

    package_dir = pathlib.Path(__file__).resolve().parents[1]
    uvicorn.run(
        "generated_adventures.webapp.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(package_dir)] if args.reload else None,
    )
