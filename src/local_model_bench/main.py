from __future__ import annotations

import sys


def main() -> int:
    try:
        try:
            from .ui import run_app
        except ImportError:
            from local_model_bench.ui import run_app
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            print("PySide6 is not installed. Run: py -3.11 -m pip install -r requirements.txt")
            return 2
        raise
    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
