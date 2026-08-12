"""E0.16 -- registered frozen-core native working-memory comparison.

This runner is intentionally read-only without ``--execute-registered``.  The
default invocation prints the checked-in ``NOT_RUN`` result.  Execution trains
only the 1,892-parameter operation gate; selected core weights and depth remain
immutable.  Resume is valid only against the complete hash-bound run identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "results" / "e0_16_native_memory_comparison.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-registered", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--run-directory",
        type=Path,
        default=ROOT / "runs" / "e0_16_registered_v1",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume and not args.execute_registered:
        raise SystemExit("--resume requires --execute-registered")
    if not args.execute_registered:
        print(json.dumps(json.loads(RESULT_PATH.read_text()), indent=2, sort_keys=True))
        return 0
    from wamrx.native_memory_comparison import run_registered_comparison

    try:
        result = run_registered_comparison(
            ROOT, args.run_directory.resolve(), resume=args.resume
        )
    except BaseException:
        state_path = args.run_directory.resolve() / "run-state.json"
        if state_path.exists():
            RESULT_PATH.write_text(state_path.read_text())
        raise
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
