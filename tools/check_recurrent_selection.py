"""Verify that the selected reasoner core matches terminal E0.14 evidence."""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wamrx.recurrent_selection import validate_selection_files  # noqa: E402


def main() -> int:
    print(json.dumps(validate_selection_files(ROOT), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
