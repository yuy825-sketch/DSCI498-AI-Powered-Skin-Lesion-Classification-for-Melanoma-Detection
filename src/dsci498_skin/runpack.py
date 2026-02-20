from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_now() -> str:
    return _utc_now()


def _git_head_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def git_head_sha(repo_root: Path) -> str | None:
    return _git_head_sha(repo_root)


def _safe_slug(name: str) -> str:
    keep = []
    for ch in name.lower():
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        elif ch.isspace():
            keep.append("-")
    slug = "".join(keep).strip("-")
    return slug or "run"


@dataclass(frozen=True)
class RunMeta:
    created_utc: str
    name: str
    cmd: str
    config_path: str
    git_head_sha: str | None
    extra: dict[str, Any]


def create_run_dir(*, runs_root: Path, name: str) -> Path:
    runs_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug = _safe_slug(name)
    outdir = runs_root / f"{ts}__{slug}"
    outdir.mkdir(parents=True, exist_ok=False)
    return outdir


def write_meta(path: Path, meta: RunMeta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)


def copy_config(config_path: Path, outdir: Path) -> Path:
    out = outdir / "config.json"
    out.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    return out
