from __future__ import annotations

import argparse
from pathlib import Path

from dsci498_skin.data.ham10000 import build_samples, load_metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/ham10000"))
    args = parser.parse_args()

    df = load_metadata(args.root)
    samples = build_samples(args.root)

    print("OK: metadata rows:", len(df))
    print("OK: resolved images:", len(samples))
    print("Columns:", ", ".join(df.columns))
    print("Class distribution (dx):")
    print(df["dx"].value_counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

