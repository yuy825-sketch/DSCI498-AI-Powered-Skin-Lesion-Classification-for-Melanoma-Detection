from __future__ import annotations

import importlib.util
import sys


def _check_import(name: str) -> None:
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise RuntimeError(f"Missing dependency: {name}")


def main() -> int:
    deps = ["torch", "torchvision", "numpy", "pandas", "sklearn", "PIL", "streamlit"]
    for dep in deps:
        _check_import(dep)

    import torch

    print("OK: imports")
    print("Python:", sys.version.split()[0])
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA current device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

