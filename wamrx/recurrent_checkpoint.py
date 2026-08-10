"""Hash-verified MLX checkpoints for registered recurrent runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .canonical import canonical_json, sha256_json
from .recurrent_model import RecurrentReasoner, parameter_count

CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_IMPLEMENTATION_VERSION = "wamrx-recurrent-checkpoint-v1"


class RecurrentCheckpointError(ValueError):
    pass


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CheckpointMetadata:
    run_id: str
    arm_id: str
    seed: int
    completed_updates: int
    schedule_hash: str
    model_implementation_hash: str
    training_config_hash: str
    split_registry_hash: str
    realized_training_flops: int
    training_examples_seen: int
    schema_version: int = CHECKPOINT_SCHEMA_VERSION
    checkpoint_implementation: str = CHECKPOINT_IMPLEMENTATION_VERSION

    def validate(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise RecurrentCheckpointError("unsupported checkpoint schema")
        if self.checkpoint_implementation != CHECKPOINT_IMPLEMENTATION_VERSION:
            raise RecurrentCheckpointError("checkpoint implementation mismatch")
        if not self.run_id or not self.arm_id:
            raise RecurrentCheckpointError("checkpoint run and arm IDs are required")
        for field in (
            "seed",
            "completed_updates",
            "realized_training_flops",
            "training_examples_seen",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RecurrentCheckpointError(
                    f"checkpoint {field} must be a nonnegative integer"
                )
        for field in (
            "schedule_hash",
            "model_implementation_hash",
            "training_config_hash",
            "split_registry_hash",
        ):
            value = getattr(self, field)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise RecurrentCheckpointError(f"checkpoint {field} is not SHA-256")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "checkpoint_implementation": self.checkpoint_implementation,
            "run_id": self.run_id,
            "arm_id": self.arm_id,
            "seed": self.seed,
            "completed_updates": self.completed_updates,
            "schedule_hash": self.schedule_hash,
            "model_implementation_hash": self.model_implementation_hash,
            "training_config_hash": self.training_config_hash,
            "split_registry_hash": self.split_registry_hash,
            "realized_training_flops": self.realized_training_flops,
            "training_examples_seen": self.training_examples_seen,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CheckpointMetadata":
        metadata = cls(
            run_id=str(value["run_id"]),
            arm_id=str(value["arm_id"]),
            seed=int(value["seed"]),
            completed_updates=int(value["completed_updates"]),
            schedule_hash=str(value["schedule_hash"]),
            model_implementation_hash=str(value["model_implementation_hash"]),
            training_config_hash=str(value["training_config_hash"]),
            split_registry_hash=str(value["split_registry_hash"]),
            realized_training_flops=int(value["realized_training_flops"]),
            training_examples_seen=int(value["training_examples_seen"]),
            schema_version=int(value["schema_version"]),
            checkpoint_implementation=str(value["checkpoint_implementation"]),
        )
        metadata.validate()
        return metadata


def _checkpoint_arrays(model: RecurrentReasoner, optimizer) -> dict[str, mx.array]:
    arrays = {
        f"model::{key}": mx.contiguous(value)
        for key, value in tree_flatten(model.trainable_parameters())
    }
    arrays.update(
        {
            f"optimizer::{key}": mx.contiguous(value)
            for key, value in tree_flatten(optimizer.state)
        }
    )
    return dict(sorted(arrays.items()))


def state_hash(model: RecurrentReasoner, optimizer) -> str:
    arrays = _checkpoint_arrays(model, optimizer)
    mx.eval(arrays)
    digest = hashlib.sha256()
    for name, array in arrays.items():
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(canonical_json(list(array.shape)).encode("ascii"))
        digest.update(memoryview(array).tobytes())
    return digest.hexdigest()


def save_checkpoint(
    path: str | Path,
    model: RecurrentReasoner,
    optimizer,
    metadata: CheckpointMetadata,
) -> dict[str, Any]:
    metadata.validate()
    if metadata.arm_id != model.arm_id:
        raise RecurrentCheckpointError("checkpoint arm does not match model")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = _checkpoint_arrays(model, optimizer)
    mx.eval(arrays)
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".safetensors",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        mx.save_safetensors(
            temporary,
            arrays,
            metadata={"wamrx_metadata": canonical_json(metadata.to_dict())},
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    report = {
        "path": str(target),
        "file_sha256": file_sha256(target),
        "state_hash": state_hash(model, optimizer),
        "bytes": target.stat().st_size,
        "parameter_count": parameter_count(model),
        "metadata_hash": sha256_json(metadata.to_dict()),
    }
    return report


def load_checkpoint(
    path: str | Path,
    model: RecurrentReasoner,
    optimizer,
    *,
    expected: CheckpointMetadata,
) -> dict[str, Any]:
    expected.validate()
    target = Path(path)
    arrays, raw_metadata = mx.load(target, return_metadata=True)
    try:
        actual = CheckpointMetadata.from_dict(
            json.loads(raw_metadata["wamrx_metadata"])
        )
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise RecurrentCheckpointError("checkpoint metadata is unreadable") from error
    if actual != expected:
        raise RecurrentCheckpointError(
            "checkpoint identity does not match the requested resume point"
        )
    model_weights = sorted(
        (name.removeprefix("model::"), value)
        for name, value in arrays.items()
        if name.startswith("model::")
    )
    optimizer_weights = sorted(
        (name.removeprefix("optimizer::"), value)
        for name, value in arrays.items()
        if name.startswith("optimizer::")
    )
    if not model_weights or not optimizer_weights:
        raise RecurrentCheckpointError("checkpoint omits model or optimizer state")
    model.load_weights(
        [
            (name, value + mx.zeros_like(value))
            for name, value in model_weights
        ],
        strict=True,
    )
    # Preserve the optimizer's parameter-tree insertion order. Reconstructing
    # from lexically sorted safetensor keys changes reduction order in gradient
    # clipping and can perturb Adam moments by a few float32 ulps after resume.
    optimizer.init(model.trainable_parameters())
    saved_optimizer = dict(optimizer_weights)
    expected_optimizer_names = [name for name, _ in tree_flatten(optimizer.state)]
    if set(saved_optimizer) != set(expected_optimizer_names):
        raise RecurrentCheckpointError("optimizer checkpoint tree does not match model")
    optimizer.state = tree_unflatten(
        [
            (
                name,
                saved_optimizer[name] + mx.zeros_like(saved_optimizer[name]),
            )
            for name in expected_optimizer_names
        ]
    )
    mx.eval(model.parameters(), optimizer.state)
    return {
        "path": str(target),
        "file_sha256": file_sha256(target),
        "state_hash": state_hash(model, optimizer),
        "bytes": target.stat().st_size,
        "parameter_count": parameter_count(model),
        "metadata_hash": sha256_json(actual.to_dict()),
        "metadata": actual.to_dict(),
    }
