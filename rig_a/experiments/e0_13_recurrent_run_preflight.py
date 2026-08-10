"""E0.13 -- train-only registered-run operational preflight.

CLAIM UNDER TEST
    A 20-update flat-recurrent training prefix is bitwise reproducible across an
    uninterrupted run and a checkpoint/resume run when model, optimizer,
    schedule, and explicit recurrent-state seeds are restored.

SCOPE
    One registered seed, the flat recurrent arm, 20 optimizer updates, batch 16.
    This experiment never loads or evaluates ID/OOD tasks and is not evidence
    for accuracy, recurrence, hierarchy, or promotion.

REGISTERED KILL CRITERIA
    R1 schedule hashes differ;
    R2 any paired per-update loss differs;
    R3 final model/optimizer state or checkpoint file hashes differ;
    R4 estimated or realized training-compute totals differ;
    R5 resume accepts an identity-mismatched checkpoint;
    R6 a non-finite loss is converted to a result rather than invalidating run;
    R7 result includes an ID/OOD/protected-region accuracy observation.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import sys
import tempfile
import time

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wamrx.canonical import sha256_json  # noqa: E402
from wamrx.recurrent_checkpoint import (  # noqa: E402
    CheckpointMetadata,
    RecurrentCheckpointError,
    file_sha256,
    load_checkpoint,
    save_checkpoint,
)
from wamrx.recurrent_model import ByteCodec, RecurrentModelConfig, RecurrentReasoner  # noqa: E402
from wamrx.recurrent_run import RunComputeRecord  # noqa: E402
from wamrx.recurrent_tasks import load_split_registry  # noqa: E402
from wamrx.recurrent_training import (  # noqa: E402
    OptimizerConfig,
    build_schedule,
    create_optimizer,
    estimated_schedule_training_flops,
    schedule_hash,
    train_schedule_segment,
    training_tasks,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAINING_CONFIG_PATH = ROOT / "contracts" / "wamrx_recurrent_training_v1.json"
SPLIT_PATH = ROOT / "contracts" / "wamrx_recurrent_splits.json"
RUN_CONTRACT_PATH = ROOT / "contracts" / "wamrx_recurrent_run_v1.json"
RESULT_PATH = ROOT / "results" / "e0_13_recurrent_run_preflight.json"
PREFLIGHT_SEED = 41001
PREFLIGHT_ARM = "flat-recurrent-v1"
PREFLIGHT_UPDATES = 20
INTERRUPTION_UPDATE = 10
PREFLIGHT_BATCH_SIZE = 16


def _implementation_hash() -> str:
    paths = (
        ROOT / "wamrx" / "recurrent_model.py",
        ROOT / "wamrx" / "recurrent_training.py",
        ROOT / "wamrx" / "recurrent_checkpoint.py",
    )
    return sha256_json(
        {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}
    )


def _metadata(
    *,
    completed_updates: int,
    frozen_schedule_hash: str,
    realized_training_flops: int,
    training_examples_seen: int,
) -> CheckpointMetadata:
    return CheckpointMetadata(
        run_id="e0.13-train-only-preflight-v1",
        arm_id=PREFLIGHT_ARM,
        seed=PREFLIGHT_SEED,
        completed_updates=completed_updates,
        schedule_hash=frozen_schedule_hash,
        model_implementation_hash=_implementation_hash(),
        training_config_hash=sha256_json(json.loads(TRAINING_CONFIG_PATH.read_text())),
        split_registry_hash=sha256_json(json.loads(SPLIT_PATH.read_text())),
        realized_training_flops=realized_training_flops,
        training_examples_seen=training_examples_seen,
    )


def run() -> dict:
    started = time.perf_counter()
    model_config = RecurrentModelConfig.load(TRAINING_CONFIG_PATH)
    optimizer_config = OptimizerConfig.load(TRAINING_CONFIG_PATH)
    registry = load_split_registry(SPLIT_PATH)
    tasks = training_tasks(registry["splits"]["train"])
    schedule = build_schedule(
        tasks,
        optimizer_config,
        seed=PREFLIGHT_SEED,
        updates=PREFLIGHT_UPDATES,
        batch_size=PREFLIGHT_BATCH_SIZE,
    )
    frozen_schedule_hash = schedule_hash(schedule)
    codec = ByteCodec(
        maximum_input_tokens=model_config.maximum_input_tokens,
        maximum_output_tokens=model_config.maximum_output_tokens,
    )

    with tempfile.TemporaryDirectory(prefix="wamrx-e0.13-") as directory:
        workspace = pathlib.Path(directory)

        mx.random.seed(PREFLIGHT_SEED)
        continuous_model = RecurrentReasoner(PREFLIGHT_ARM, model_config)
        continuous_optimizer = create_optimizer(
            optimizer_config, total_updates=PREFLIGHT_UPDATES
        )
        continuous = train_schedule_segment(
            continuous_model,
            codec,
            tasks,
            schedule,
            optimizer_config,
            optimizer=continuous_optimizer,
            start_update=0,
            stop_update=PREFLIGHT_UPDATES,
        )
        continuous_metadata = _metadata(
            completed_updates=PREFLIGHT_UPDATES,
            frozen_schedule_hash=frozen_schedule_hash,
            realized_training_flops=continuous["realized_training_flops"],
            training_examples_seen=continuous["examples_seen"],
        )
        continuous_checkpoint = save_checkpoint(
            workspace / "continuous-final.safetensors",
            continuous_model,
            continuous_optimizer,
            continuous_metadata,
        )

        mx.random.seed(PREFLIGHT_SEED)
        interrupted_model = RecurrentReasoner(PREFLIGHT_ARM, model_config)
        interrupted_optimizer = create_optimizer(
            optimizer_config, total_updates=PREFLIGHT_UPDATES
        )
        prefix = train_schedule_segment(
            interrupted_model,
            codec,
            tasks,
            schedule,
            optimizer_config,
            optimizer=interrupted_optimizer,
            start_update=0,
            stop_update=INTERRUPTION_UPDATE,
        )
        prefix_metadata = _metadata(
            completed_updates=INTERRUPTION_UPDATE,
            frozen_schedule_hash=frozen_schedule_hash,
            realized_training_flops=prefix["realized_training_flops"],
            training_examples_seen=prefix["examples_seen"],
        )
        prefix_path = workspace / "interrupted-update-0010.safetensors"
        save_checkpoint(
            prefix_path,
            interrupted_model,
            interrupted_optimizer,
            prefix_metadata,
        )

        mx.random.seed(999999)
        resumed_model = RecurrentReasoner(PREFLIGHT_ARM, model_config)
        resumed_optimizer = create_optimizer(
            optimizer_config, total_updates=PREFLIGHT_UPDATES
        )
        identity_mismatch_rejected = False
        try:
            load_checkpoint(
                prefix_path,
                resumed_model,
                resumed_optimizer,
                expected=dataclasses.replace(prefix_metadata, seed=41002),
            )
        except RecurrentCheckpointError:
            identity_mismatch_rejected = True
        load_checkpoint(
            prefix_path,
            resumed_model,
            resumed_optimizer,
            expected=prefix_metadata,
        )
        suffix = train_schedule_segment(
            resumed_model,
            codec,
            tasks,
            schedule,
            optimizer_config,
            optimizer=resumed_optimizer,
            start_update=INTERRUPTION_UPDATE,
            stop_update=PREFLIGHT_UPDATES,
        )
        resumed_realized_flops = (
            prefix["realized_training_flops"] + suffix["realized_training_flops"]
        )
        resumed_examples = prefix["examples_seen"] + suffix["examples_seen"]
        resumed_metadata = _metadata(
            completed_updates=PREFLIGHT_UPDATES,
            frozen_schedule_hash=frozen_schedule_hash,
            realized_training_flops=resumed_realized_flops,
            training_examples_seen=resumed_examples,
        )
        resumed_checkpoint = save_checkpoint(
            workspace / "resumed-final.safetensors",
            resumed_model,
            resumed_optimizer,
            resumed_metadata,
        )

        estimated_training_flops = estimated_schedule_training_flops(
            continuous_model, schedule
        )
        continuous_compute = RunComputeRecord(
            run_id="e0.13-train-only-preflight-v1",
            arm_id=PREFLIGHT_ARM,
            seed=PREFLIGHT_SEED,
            status="COMPLETE",
            parameter_count=continuous_checkpoint["parameter_count"],
            schedule_hash=frozen_schedule_hash,
            estimated_training_flops=estimated_training_flops,
            realized_training_flops=continuous["realized_training_flops"],
            estimated_inference_flops=0,
            realized_inference_flops=0,
            training_examples_planned=PREFLIGHT_UPDATES * PREFLIGHT_BATCH_SIZE,
            training_examples_seen=continuous["examples_seen"],
            optimizer_updates_planned=PREFLIGHT_UPDATES,
            optimizer_updates_completed=PREFLIGHT_UPDATES,
            evaluation_examples_planned=0,
            evaluation_examples_completed=0,
        ).to_dict()

        checks = {
            "R1_schedule_hash_exact": (
                continuous["schedule_hash"]
                == prefix["schedule_hash"]
                == suffix["schedule_hash"]
                == frozen_schedule_hash
            ),
            "R2_per_update_losses_exact": (
                continuous["losses"] == prefix["losses"] + suffix["losses"]
            ),
            "R3_final_model_optimizer_state_exact": (
                continuous_checkpoint["state_hash"]
                == resumed_checkpoint["state_hash"]
            ),
            "R3_final_checkpoint_file_exact": (
                continuous_checkpoint["file_sha256"]
                == resumed_checkpoint["file_sha256"]
            ),
            "R4_compute_accounting_exact": (
                estimated_training_flops
                == continuous["realized_training_flops"]
                == resumed_realized_flops
            ),
            "R4_example_and_update_counts_exact": (
                continuous["examples_seen"] == resumed_examples
                and continuous["updates"] == PREFLIGHT_UPDATES
                and prefix["updates"] + suffix["updates"] == PREFLIGHT_UPDATES
            ),
            "R5_identity_mismatched_resume_rejected": identity_mismatch_rejected,
            "R6_all_losses_finite": all(
                math.isfinite(value)
                for value in continuous["losses"] + prefix["losses"] + suffix["losses"]
            ),
            "R7_train_only_no_evaluation_metrics": True,
        }
        checkpoint_bytes = continuous_checkpoint["bytes"]

    elapsed = time.perf_counter() - started
    return {
        "experiment": "E0.13",
        "claim": "registered recurrent training is bitwise checkpoint/resume reproducible at the 20-update operational-preflight scope",
        "scope": "train-only; one seed; flat recurrent arm; no ID/OOD or protected-region evaluation",
        "model_comparison_status": "NOT_RUN",
        "run_contract_hash": sha256_json(json.loads(RUN_CONTRACT_PATH.read_text())),
        "seed": PREFLIGHT_SEED,
        "arm_id": PREFLIGHT_ARM,
        "updates": PREFLIGHT_UPDATES,
        "interruption_update": INTERRUPTION_UPDATE,
        "batch_size": PREFLIGHT_BATCH_SIZE,
        "schedule_hash": frozen_schedule_hash,
        "continuous_compute": continuous_compute,
        "checkpoint_bytes": checkpoint_bytes,
        "wall_clock_seconds": elapsed,
        "projected_final_checkpoint_storage_bytes": checkpoint_bytes * 15,
        "projected_periodic_checkpoint_storage_bytes": (
            checkpoint_bytes * 15 * (2000 // 50)
        ),
        "projected_primary_training_wall_clock_seconds_serial_lower_bound": (
            elapsed * (2000 / PREFLIGHT_UPDATES) * 15 / 2
        ),
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }


def main() -> int:
    result = run()
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
