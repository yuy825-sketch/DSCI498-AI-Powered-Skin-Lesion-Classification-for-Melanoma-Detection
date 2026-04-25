from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_subprocess(args: list[str]) -> int:
    completed = subprocess.run(args, cwd=ROOT)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convenience entrypoint for the DSCI 498 skin lesion classification project."
    )
    subparsers = parser.add_subparsers(dest="command")

    smoke_parser = subparsers.add_parser("smoke", help="Run the package smoke test.")

    train_parser = subparsers.add_parser("train", help="Delegate to train.py.")
    train_parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra arguments passed to train.py")

    vae_parser = subparsers.add_parser("train-vae", help="Delegate to train_vae.py.")
    vae_parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra arguments passed to train_vae.py")

    demo_parser = subparsers.add_parser("demo", help="Launch the Streamlit demo app.")
    demo_parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra arguments passed to streamlit")

    args = parser.parse_args()

    if args.command == "smoke":
        return run_subprocess([sys.executable, "-m", "dsci498_skin.smoke"])

    if args.command == "train":
        return run_subprocess([sys.executable, "train.py", *args.extra])

    if args.command == "train-vae":
        return run_subprocess([sys.executable, "train_vae.py", *args.extra])

    if args.command == "demo":
        return run_subprocess(["streamlit", "run", "app/app.py", *args.extra])

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
