"""Gradient/runtime smoke test for the three frozen recurrent model arms.

This is not a registered performance experiment. It uses two optimizer updates
per arm only to prove that the parameter-matched MLX graphs train, decode, and
emit compute accounting under one shared schedule.
"""

from __future__ import annotations

import json
import pathlib
import sys

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wamrx.recurrent_model import (  # noqa: E402
    ARM_IDS,
    ByteCodec,
    RecurrentModelConfig,
    RecurrentReasoner,
)
from wamrx.recurrent_tasks import load_split_registry  # noqa: E402
from wamrx.recurrent_training import (  # noqa: E402
    OptimizerConfig,
    build_schedule,
    compute_record_for_batch,
    evaluate_exact,
    train_updates,
    training_tasks,
    validate_shared_schedule,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING_CONFIG = ROOT / "contracts" / "wamrx_recurrent_training_v1.json"
SPLITS = ROOT / "contracts" / "wamrx_recurrent_splits.json"


def run() -> dict:
    model_config = RecurrentModelConfig.load(TRAINING_CONFIG)
    optimizer_config = OptimizerConfig.load(TRAINING_CONFIG)
    registry = load_split_registry(SPLITS)
    tasks = training_tasks(registry["splits"]["train"])
    schedule = build_schedule(
        tasks,
        optimizer_config,
        seed=optimizer_config.paired_seeds[0],
        updates=2,
        batch_size=2,
    )
    codec = ByteCodec(
        maximum_input_tokens=model_config.maximum_input_tokens,
        maximum_output_tokens=model_config.maximum_output_tokens,
    )
    reports = []
    evaluations = {}
    accounting = {}
    sample = list(tasks[:2])
    for arm_id in ARM_IDS:
        mx.random.seed(optimizer_config.paired_seeds[0])
        model = RecurrentReasoner(arm_id, model_config)
        report = train_updates(
            model,
            codec,
            tasks,
            schedule,
            optimizer_config,
        )
        reports.append(report)
        evaluations[arm_id] = evaluate_exact(
            model,
            codec,
            sample,
            macro_steps=2,
            micro_steps=1,
        )
        accounting[arm_id] = compute_record_for_batch(
            model,
            sample,
            macro_steps=2,
            micro_steps=1,
            training_examples_seen=report["examples_seen"],
            optimizer_updates=report["updates"],
        ).to_dict()
    validate_shared_schedule(reports)
    return {
        "scope": "unregistered two-update runtime smoke test; not a model comparison",
        "training_reports": reports,
        "sample_evaluations": evaluations,
        "compute_accounting": accounting,
        "passed": all(report["updates"] == 2 for report in reports),
    }


def main() -> int:
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
