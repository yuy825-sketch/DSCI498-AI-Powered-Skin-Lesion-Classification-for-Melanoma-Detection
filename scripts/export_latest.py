from __future__ import annotations

import argparse
from pathlib import Path

from export_results import export_run


def _latest_run_dir(runs_root: Path) -> Path:
    candidates = [p for p in runs_root.iterdir() if p.is_dir() and (p / "metrics.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No completed runs found under {runs_root}")
    return sorted(candidates)[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    latest = _latest_run_dir(args.runs_root)
    print("Latest run:", latest)
    export_run(run_dir=latest, out_dir=args.out_dir)
    print("Wrote:", args.out_dir / "summary.md")
    print("Wrote:", args.out_dir / "confusion_matrix.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

