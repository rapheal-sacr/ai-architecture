"""Optional MLX gate and frozen-core loader for E0.16.

The selected 8.39M-parameter reasoner is loaded only from the five registered
E0.14 checkpoints and never appears in a gradient tree.  The sole trainable
module is one 472-by-4 affine operation classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from .canonical import canonical_json, sha256_json
from .native_memory_run import (
    GateBatch,
    GateExample,
    OPERATION_ORDER,
    gate_schedule_hash,
)
from .recurrent_checkpoint import (
    CheckpointMetadata,
    file_sha256,
    load_checkpoint,
)
from .recurrent_model import (
    ByteCodec,
    RecurrentModelConfig,
    RecurrentReasoner,
    parameter_count,
)
from .recurrent_training import OptimizerConfig, create_optimizer


GATE_IMPLEMENTATION_VERSION = "wamrx-minimal-operation-gate-v1"
GATE_CHECKPOINT_VERSION = "wamrx-native-memory-gate-checkpoint-v1"
GATE_PARAMETER_COUNT = 472 * 4 + 4
GATE_FORWARD_FLOPS_PER_EXAMPLE = 2 * 472 * 4 + 4
GATE_BACKWARD_MULTIPLIER = 2
ADAMW_FLOPS_PER_PARAMETER = 17


class NativeMemoryModelError(ValueError):
    pass


def _array_tree_hash(values: Mapping[str, mx.array]) -> str:
    ordered = dict(sorted(values.items()))
    mx.eval(ordered)
    digest = hashlib.sha256()
    for name, array in ordered.items():
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(canonical_json(list(array.shape)).encode("ascii"))
        digest.update(memoryview(array).tobytes())
    return digest.hexdigest()


def model_parameter_hash(model: RecurrentReasoner) -> str:
    return _array_tree_hash(dict(tree_flatten(model.trainable_parameters())))


class MinimalOperationGate(nn.Module):
    def __init__(self, hidden_dimensions: int = 472) -> None:
        super().__init__()
        if hidden_dimensions != 472:
            raise NativeMemoryModelError("registered gate input must have 472 dimensions")
        self.operation_logits = nn.Linear(hidden_dimensions, len(OPERATION_ORDER))
        if parameter_count(self) != GATE_PARAMETER_COUNT:
            raise NativeMemoryModelError("minimal gate parameter count drifted")

    def __call__(self, features: mx.array) -> mx.array:
        if features.ndim != 2 or features.shape[-1] != 472:
            raise NativeMemoryModelError("gate features must have shape [batch, 472]")
        return self.operation_logits(features)

    def decide(self, features: mx.array) -> tuple[str, ...]:
        logits = self(features)
        mx.eval(logits)
        return tuple(OPERATION_ORDER[index] for index in mx.argmax(logits, axis=-1).tolist())


@dataclass(frozen=True)
class FrozenCoreIdentity:
    seed: int
    path: str
    file_sha256: str
    metadata_hash: str
    state_hash: str
    model_parameter_hash: str
    parameter_count: int
    macro_depth: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "path": self.path,
            "file_sha256": self.file_sha256,
            "metadata_hash": self.metadata_hash,
            "state_hash": self.state_hash,
            "model_parameter_hash": self.model_parameter_hash,
            "parameter_count": self.parameter_count,
            "macro_depth": self.macro_depth,
        }


def _registered_checkpoint(root: Path, seed: int) -> dict[str, Any]:
    manifest = json.loads(
        (root / "contracts" / "wamrx_native_memory_core_checkpoints_v1.json").read_text()
    )
    matches = [row for row in manifest["checkpoints"] if int(row["seed"]) == seed]
    if len(matches) != 1:
        raise NativeMemoryModelError(f"seed {seed} is not uniquely registered")
    return {**matches[0], "parameter_count": int(manifest["parameter_count"])}


def load_frozen_core(root: Path, seed: int) -> tuple[RecurrentReasoner, FrozenCoreIdentity]:
    row = _registered_checkpoint(root, seed)
    path = root / row["relocation_path"]
    if not path.is_file():
        raise NativeMemoryModelError(f"registered core checkpoint is unavailable: {path}")
    if path.stat().st_size != int(row["bytes"]) or file_sha256(path) != row["file_sha256"]:
        raise NativeMemoryModelError("registered core checkpoint bytes or hash drifted")

    arrays, raw_metadata = mx.load(path, return_metadata=True)
    try:
        metadata = CheckpointMetadata.from_dict(
            json.loads(raw_metadata["wamrx_metadata"])
        )
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise NativeMemoryModelError("registered core metadata is unreadable") from error
    if sha256_json(metadata.to_dict()) != row["metadata_hash"]:
        raise NativeMemoryModelError("registered core metadata hash drifted")
    if (
        metadata.seed != seed
        or metadata.arm_id != "fixed-depth-v1"
        or metadata.completed_updates != 2000
    ):
        raise NativeMemoryModelError("registered core checkpoint identity drifted")
    del arrays

    training_path = root / "contracts" / "wamrx_recurrent_training_v1.json"
    model_config = RecurrentModelConfig.load(training_path)
    optimizer_config = OptimizerConfig.load(training_path)
    mx.random.seed(seed)
    model = RecurrentReasoner("fixed-depth-v1", model_config)
    optimizer = create_optimizer(
        optimizer_config,
        total_updates=optimizer_config.optimizer_updates,
    )
    report = load_checkpoint(path, model, optimizer, expected=metadata)
    if report["state_hash"] != row["state_hash"]:
        raise NativeMemoryModelError("registered core model-plus-optimizer state hash drifted")
    if report["parameter_count"] != int(row["parameter_count"]):
        raise NativeMemoryModelError("registered core parameter count drifted")
    identity = FrozenCoreIdentity(
        seed=seed,
        path=str(path),
        file_sha256=report["file_sha256"],
        metadata_hash=report["metadata_hash"],
        state_hash=report["state_hash"],
        model_parameter_hash=model_parameter_hash(model),
        parameter_count=report["parameter_count"],
        macro_depth=4,
    )
    del optimizer
    return model, identity


def gate_features(
    model: RecurrentReasoner,
    codec: ByteCodec,
    examples: tuple[GateExample, ...],
) -> tuple[mx.array, mx.array]:
    tokens = []
    targets = []
    operation_index = {operation: index for index, operation in enumerate(OPERATION_ORDER)}
    for example in examples:
        encoded, _ = codec.encode_problem(example.input)
        tokens.append(encoded)
        targets.append(operation_index[example.target_operation])
    token_array = mx.array(tokens, dtype=mx.int32)
    features = model.encode(token_array)
    mx.eval(features)
    return features, mx.array(targets, dtype=mx.int32)


def _loss(gate: MinimalOperationGate, features: mx.array, targets: mx.array) -> mx.array:
    return nn.losses.cross_entropy(gate(features), targets, reduction="mean")


def create_gate_optimizer(contract: dict[str, Any]):
    training = contract["gate_training"]
    return optim.AdamW(
        learning_rate=float(training["learning_rate"]),
        betas=list(map(float, training["betas"])),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )


def train_gate_segment(
    gate: MinimalOperationGate,
    optimizer,
    *,
    feature_by_id: Mapping[str, mx.array],
    target_by_id: Mapping[str, int],
    schedule: tuple[GateBatch, ...],
    gradient_clip_norm: float,
    start_update: int,
    stop_update: int,
) -> dict[str, Any]:
    if not 0 <= start_update <= stop_update <= len(schedule):
        raise NativeMemoryModelError("gate training segment is outside its schedule")
    loss_and_grad = nn.value_and_grad(gate, _loss)
    losses: list[float] = []
    examples_seen = 0
    for expected_update, batch in enumerate(
        schedule[start_update:stop_update], start=start_update
    ):
        if batch.update != expected_update:
            raise NativeMemoryModelError("gate schedule updates are not contiguous")
        features = mx.stack([feature_by_id[item] for item in batch.example_ids])
        targets = mx.array([target_by_id[item] for item in batch.example_ids], dtype=mx.int32)
        loss, gradients = loss_and_grad(gate, features, targets)
        gradients, gradient_norm = optim.clip_grad_norm(gradients, gradient_clip_norm)
        optimizer.update(gate, gradients)
        optimizer.state = tree_unflatten(
            [(name, value + mx.zeros_like(value)) for name, value in tree_flatten(optimizer.state)]
        )
        gate.load_weights(
            [(name, value + mx.zeros_like(value)) for name, value in tree_flatten(gate.trainable_parameters())],
            strict=True,
        )
        mx.eval(loss, gradient_norm, gate.parameters(), optimizer.state)
        value = float(loss)
        if not math.isfinite(value):
            raise NativeMemoryModelError("gate training produced a non-finite loss")
        losses.append(value)
        examples_seen += len(batch.example_ids)
    return {
        "start_update": start_update,
        "stop_update": stop_update,
        "updates": stop_update - start_update,
        "examples_seen": examples_seen,
        "losses": losses,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "realized_gate_training_flops": sum(
            len(batch.example_ids)
            * GATE_FORWARD_FLOPS_PER_EXAMPLE
            * (1 + GATE_BACKWARD_MULTIPLIER)
            + ADAMW_FLOPS_PER_PARAMETER * GATE_PARAMETER_COUNT
            for batch in schedule[start_update:stop_update]
        ),
    }


@dataclass(frozen=True)
class GateCheckpointMetadata:
    seed: int
    completed_updates: int
    examples_seen: int
    schedule_hash: str
    run_contract_hash: str
    core_file_sha256: str
    gate_parameter_count: int = GATE_PARAMETER_COUNT
    checkpoint_version: str = GATE_CHECKPOINT_VERSION

    def validate(self) -> None:
        if self.checkpoint_version != GATE_CHECKPOINT_VERSION:
            raise NativeMemoryModelError("gate checkpoint version drifted")
        if self.gate_parameter_count != GATE_PARAMETER_COUNT:
            raise NativeMemoryModelError("gate checkpoint parameter count drifted")
        if min(self.seed, self.completed_updates, self.examples_seen) < 0:
            raise NativeMemoryModelError("gate checkpoint counters must be nonnegative")
        for value in (self.schedule_hash, self.run_contract_hash, self.core_file_sha256):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise NativeMemoryModelError("gate checkpoint contains a non-SHA-256 identity")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "checkpoint_version": self.checkpoint_version,
            "seed": self.seed,
            "completed_updates": self.completed_updates,
            "examples_seen": self.examples_seen,
            "schedule_hash": self.schedule_hash,
            "run_contract_hash": self.run_contract_hash,
            "core_file_sha256": self.core_file_sha256,
            "gate_parameter_count": self.gate_parameter_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GateCheckpointMetadata":
        metadata = cls(
            seed=int(value["seed"]),
            completed_updates=int(value["completed_updates"]),
            examples_seen=int(value["examples_seen"]),
            schedule_hash=str(value["schedule_hash"]),
            run_contract_hash=str(value["run_contract_hash"]),
            core_file_sha256=str(value["core_file_sha256"]),
            gate_parameter_count=int(value["gate_parameter_count"]),
            checkpoint_version=str(value["checkpoint_version"]),
        )
        metadata.validate()
        return metadata


def _gate_checkpoint_arrays(gate: MinimalOperationGate, optimizer) -> dict[str, mx.array]:
    arrays = {
        f"gate::{name}": mx.contiguous(value)
        for name, value in tree_flatten(gate.trainable_parameters())
    }
    arrays.update(
        {
            f"optimizer::{name}": mx.contiguous(value)
            for name, value in tree_flatten(optimizer.state)
        }
    )
    return dict(sorted(arrays.items()))


def gate_state_hash(gate: MinimalOperationGate, optimizer) -> str:
    return _array_tree_hash(_gate_checkpoint_arrays(gate, optimizer))


def save_gate_checkpoint(
    path: Path,
    gate: MinimalOperationGate,
    optimizer,
    metadata: GateCheckpointMetadata,
) -> dict[str, Any]:
    metadata.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _gate_checkpoint_arrays(gate, optimizer)
    mx.eval(arrays)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".safetensors",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        mx.save_safetensors(
            temporary,
            arrays,
            metadata={"wamrx_gate_metadata": canonical_json(metadata.to_dict())},
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "file_sha256": file_sha256(path),
        "metadata_hash": sha256_json(metadata.to_dict()),
        "state_hash": gate_state_hash(gate, optimizer),
    }


def load_gate_checkpoint(
    path: Path,
    gate: MinimalOperationGate,
    optimizer,
    *,
    expected: GateCheckpointMetadata,
) -> dict[str, Any]:
    arrays, raw_metadata = mx.load(path, return_metadata=True)
    try:
        actual = GateCheckpointMetadata.from_dict(
            json.loads(raw_metadata["wamrx_gate_metadata"])
        )
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise NativeMemoryModelError("gate checkpoint metadata is unreadable") from error
    if actual != expected:
        raise NativeMemoryModelError("gate checkpoint identity mismatch")
    gate_weights = sorted(
        (name.removeprefix("gate::"), value)
        for name, value in arrays.items()
        if name.startswith("gate::")
    )
    optimizer_weights = {
        name.removeprefix("optimizer::"): value
        for name, value in arrays.items()
        if name.startswith("optimizer::")
    }
    gate.load_weights(gate_weights, strict=True)
    optimizer.init(gate.trainable_parameters())
    expected_names = [name for name, _ in tree_flatten(optimizer.state)]
    if set(optimizer_weights) != set(expected_names):
        raise NativeMemoryModelError("gate optimizer checkpoint tree mismatch")
    optimizer.state = tree_unflatten(
        [(name, optimizer_weights[name] + mx.zeros_like(optimizer_weights[name])) for name in expected_names]
    )
    mx.eval(gate.parameters(), optimizer.state)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "file_sha256": file_sha256(path),
        "metadata_hash": sha256_json(actual.to_dict()),
        "state_hash": gate_state_hash(gate, optimizer),
    }


def feature_maps(
    model: RecurrentReasoner,
    codec: ByteCodec,
    examples: tuple[GateExample, ...],
) -> tuple[dict[str, mx.array], dict[str, int]]:
    features, targets = gate_features(model, codec, examples)
    return (
        {example.example_id: features[index] for index, example in enumerate(examples)},
        {example.example_id: int(targets[index].item()) for index, example in enumerate(examples)},
    )


def run_gate_preflight(
    root: Path,
    *,
    seed: int,
    examples: tuple[GateExample, ...],
    schedule: tuple[GateBatch, ...],
) -> dict[str, Any]:
    contract = json.loads(
        (root / "contracts" / "wamrx_native_memory_run_v1.json").read_text()
    )
    contract_hash = sha256_json(contract)
    core, identity = load_frozen_core(root, seed)
    before = model_parameter_hash(core)
    config = core.config
    codec = ByteCodec(
        maximum_input_tokens=config.maximum_input_tokens,
        maximum_output_tokens=config.maximum_output_tokens,
    )
    feature_by_id, target_by_id = feature_maps(core, codec, examples)
    training = contract["gate_training"]

    def new_pair():
        mx.random.seed(seed + int(training["gate_initialization_seed_offset"]))
        gate = MinimalOperationGate(config.hidden_dimensions)
        optimizer = create_gate_optimizer(contract)
        return gate, optimizer

    continuous_gate, continuous_optimizer = new_pair()
    continuous = train_gate_segment(
        continuous_gate,
        continuous_optimizer,
        feature_by_id=feature_by_id,
        target_by_id=target_by_id,
        schedule=schedule,
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        start_update=0,
        stop_update=2,
    )
    continuous_hash = gate_state_hash(continuous_gate, continuous_optimizer)

    interrupted_gate, interrupted_optimizer = new_pair()
    first = train_gate_segment(
        interrupted_gate,
        interrupted_optimizer,
        feature_by_id=feature_by_id,
        target_by_id=target_by_id,
        schedule=schedule,
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        start_update=0,
        stop_update=1,
    )
    metadata = GateCheckpointMetadata(
        seed=seed,
        completed_updates=1,
        examples_seen=len(schedule[0].example_ids),
        schedule_hash=gate_schedule_hash(schedule),
        run_contract_hash=contract_hash,
        core_file_sha256=identity.file_sha256,
    )
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "gate-update-0001.safetensors"
        saved = save_gate_checkpoint(
            checkpoint_path,
            interrupted_gate,
            interrupted_optimizer,
            metadata,
        )
        resumed_gate, resumed_optimizer = new_pair()
        loaded = load_gate_checkpoint(
            checkpoint_path,
            resumed_gate,
            resumed_optimizer,
            expected=metadata,
        )
        second = train_gate_segment(
            resumed_gate,
            resumed_optimizer,
            feature_by_id=feature_by_id,
            target_by_id=target_by_id,
            schedule=schedule,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            start_update=1,
            stop_update=2,
        )
        resumed_hash = gate_state_hash(resumed_gate, resumed_optimizer)
    after = model_parameter_hash(core)
    if before != after or before != identity.model_parameter_hash:
        raise NativeMemoryModelError("preflight mutated the frozen core")
    if continuous_hash != resumed_hash:
        raise NativeMemoryModelError("gate checkpoint/resume is not bitwise exact")
    if continuous["losses"] != [*first["losses"], *second["losses"]]:
        raise NativeMemoryModelError("gate resumed loss sequence drifted")
    return {
        "status": "PASS",
        "seed": seed,
        "accuracy_metrics_read": 0,
        "gate_parameter_count": parameter_count(continuous_gate),
        "core_identity": identity.to_dict(),
        "core_hash_before": before,
        "core_hash_after": after,
        "schedule_hash": gate_schedule_hash(schedule),
        "continuous_state_hash": continuous_hash,
        "resumed_state_hash": resumed_hash,
        "continuous_losses": continuous["losses"],
        "resumed_losses": [*first["losses"], *second["losses"]],
        "checkpoint_saved": saved,
        "checkpoint_loaded": loaded,
    }
