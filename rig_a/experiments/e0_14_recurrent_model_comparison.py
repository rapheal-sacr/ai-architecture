"""E0.14 -- registered five-seed recurrent model comparison.

This file is intentionally read-only without ``--execute-registered``: its
default invocation prints the checked-in terminal result. The frozen budget has
been consumed and the terminal decision is ``COMPLETE_RETAIN_FIXED``.
Interrupted registered runs may use ``--resume``; seed substitution is
forbidden.

PRIMARY CLAIM
    Under equal updates, equal examples, shared data/interface/optimizer, and a
    common maximum inference-FLOP cap, determine whether fixed depth, flat
    recurrence, or hierarchical recurrence satisfies the preregistered gate.

INTERPRETATION LIMIT
    Realized compute is recorded rather than assumed equal. A gain that fails
    the compute-normalized secondary is classified as compute scaling, not
    architectural efficiency.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wamrx.recurrent_comparison import run_registered_comparison  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "results" / "e0_14_recurrent_model_comparison.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-registered",
        action="store_true",
        help="consume the frozen five-seed, three-arm training/evaluation budget",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only from hash-verified checkpoints and journals",
    )
    parser.add_argument(
        "--run-directory",
        type=pathlib.Path,
        default=ROOT / "runs" / "e0_14_registered_v1",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute_registered:
        current = json.loads(RESULT_PATH.read_text())
        print(json.dumps(current, indent=2, sort_keys=True))
        return 0
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
