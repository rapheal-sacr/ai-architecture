"""Validate the frozen E0.16 run registration without importing MLX."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wamrx.native_memory_run import validate_run_registration  # noqa: E402


def main() -> int:
    print(json.dumps(validate_run_registration(ROOT), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
