"""Frozen-data MLX training and evaluation helpers for recurrent reasoners."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from .canonical import sha256_json
from .recurrent import ComputeBudget, ComputeRecord
from .recurrent_model import (
    ARM_IDS,
    ByteCodec,
    RecurrentModelConfig,
    RecurrentReasoner,
    batch_arrays,
    decode_batch,
    parameter_count,
    token_loss,
)
from .recurrent_tasks import FAMILIES, RecurrentTask, generate_family

TRAINING_IMPLEMENTATION_VERSION = "wamrx-recurrent-training-v1"


class RecurrentTrainingError(ValueError):
    pass


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    gradient_clip_norm: float
    batch_size: int
    optimizer_updates: int
    warmup_updates: int
    minimum_learning_rate_ratio: float
    intermediate_state_loss_weight: float
    halt_binary_cross_entropy_weight: float
    initial_state_zero_probability: float
    initial_state_standard_deviation: float
    train_macro_depths: tuple[int, ...]
    train_micro_depths: tuple[int, ...]
    paired_seeds: tuple[int, ...]

    @classmethod
    def load(cls, path: str | Path) -> "OptimizerConfig":
        value = json.loads(Path(path).read_text())
        optimizer = value["optimizer"]
        loss = value["loss"]
        depth = value["depth_sampling"]
        initial = value["initial_state"]
        config = cls(
            learning_rate=float(optimizer["learning_rate"]),
            betas=tuple(map(float, optimizer["betas"])),
            epsilon=float(optimizer["epsilon"]),
            weight_decay=float(optimizer["weight_decay"]),
            gradient_clip_norm=float(optimizer["gradient_clip_norm"]),
            batch_size=int(optimizer["batch_size"]),
            optimizer_updates=int(optimizer["optimizer_updates"]),
            warmup_updates=int(optimizer["warmup_updates"]),
            minimum_learning_rate_ratio=float(
                optimizer["minimum_learning_rate_ratio"]
            ),
            intermediate_state_loss_weight=float(
                loss["intermediate_state_loss_weight"]
            ),
            halt_binary_cross_entropy_weight=float(
                loss["halt_binary_cross_entropy_weight"]
            ),
            initial_state_zero_probability=float(initial["zero_probability"]),
            initial_state_standard_deviation=float(
                initial["gaussian_standard_deviation"]
            ),
            train_macro_depths=tuple(map(int, depth["train_macro_depths"])),
            train_micro_depths=tuple(map(int, depth["train_micro_depths"])),
            paired_seeds=tuple(map(int, value["paired_seeds"])),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if len(self.betas) != 2 or not all(0.0 < value < 1.0 for value in self.betas):
            raise RecurrentTrainingError("AdamW betas must contain two values in (0, 1)")
        if min(
            self.learning_rate,
            self.epsilon,
            self.gradient_clip_norm,
            self.batch_size,
            self.optimizer_updates,
        ) <= 0:
            raise RecurrentTrainingError("optimizer scales and counts must be positive")
        if not 0.0 <= self.initial_state_zero_probability <= 1.0:
            raise RecurrentTrainingError("initial-state mixture probability is invalid")
        if len(set(self.train_macro_depths)) < 2 or len(set(self.train_micro_depths)) < 2:
            raise RecurrentTrainingError("training depth distributions must be randomized")
        if len(self.paired_seeds) < 5:
            raise RecurrentTrainingError("at least five paired seeds are required")


@dataclass(frozen=True)
class ScheduledBatch:
    update: int
    task_ids: tuple[str, ...]
    macro_steps: int
    micro_steps: int
    initial_state_seed: int
    use_zero_initial_state: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "update": self.update,
            "task_ids": list(self.task_ids),
            "macro_steps": self.macro_steps,
            "micro_steps": self.micro_steps,
            "initial_state_seed": self.initial_state_seed,
            "use_zero_initial_state": self.use_zero_initial_state,
        }


def training_tasks(split_spec: dict[str, Any]) -> tuple[RecurrentTask, ...]:
    tasks = tuple(
        task
        for family in FAMILIES
        for task in generate_family("train", family, split_spec)
    )
    return tuple(sorted(tasks, key=lambda task: task.task_id))


def build_schedule(
    tasks: tuple[RecurrentTask, ...],
    config: OptimizerConfig,
    *,
    seed: int,
    updates: int | None = None,
    batch_size: int | None = None,
) -> tuple[ScheduledBatch, ...]:
    rng = random.Random(seed)
    update_count = config.optimizer_updates if updates is None else updates
    size = config.batch_size if batch_size is None else batch_size
    if update_count < 1 or size < 1:
        raise RecurrentTrainingError("schedule update and batch counts must be positive")
    order = list(range(len(tasks)))
    cursor = len(order)
    batches = []
    for update in range(update_count):
        if cursor + size > len(order):
            rng.shuffle(order)
            cursor = 0
        indices = order[cursor : cursor + size]
        cursor += size
        batches.append(
            ScheduledBatch(
                update=update,
                task_ids=tuple(tasks[index].task_id for index in indices),
                macro_steps=rng.choice(config.train_macro_depths),
                micro_steps=rng.choice(config.train_micro_depths),
                initial_state_seed=rng.randrange(2**31),
                use_zero_initial_state=(
                    rng.random() < config.initial_state_zero_probability
                ),
            )
        )
    return tuple(batches)


def schedule_hash(schedule: tuple[ScheduledBatch, ...]) -> str:
    return sha256_json([batch.to_dict() for batch in schedule])


def _learning_rate_schedule(config: OptimizerConfig, updates: int):
    warmup = min(config.warmup_updates, max(1, updates // 10))
    peak = config.learning_rate
    floor = peak * config.minimum_learning_rate_ratio
    if updates <= warmup:
        return optim.linear_schedule(floor, peak, updates)
    return optim.join_schedules(
        [
            optim.linear_schedule(floor, peak, warmup),
            optim.cosine_decay(peak, updates - warmup, end=floor),
        ],
        [warmup],
    )


def create_optimizer(config: OptimizerConfig, *, total_updates: int):
    if total_updates < 1:
        raise RecurrentTrainingError("optimizer schedule must contain an update")
    return optim.AdamW(
        learning_rate=_learning_rate_schedule(config, total_updates),
        betas=list(config.betas),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )


def effective_depths(
    model: RecurrentReasoner,
    batch: ScheduledBatch,
) -> tuple[int, int]:
    if model.arm_id == "fixed-depth-v1":
        return model.config.fixed_reasoning_blocks, 1
    if model.arm_id == "flat-recurrent-v1":
        return batch.macro_steps, 1
    return batch.macro_steps, batch.micro_steps


def scheduled_batch_flops(
    model: RecurrentReasoner,
    batch: ScheduledBatch,
) -> int:
    macro_steps, micro_steps = effective_depths(model, batch)
    return model.estimated_training_update_flops(
        batch_size=len(batch.task_ids),
        input_tokens=model.config.maximum_input_tokens,
        macro_steps=macro_steps,
        micro_steps=micro_steps,
    )


def estimated_schedule_training_flops(
    model: RecurrentReasoner,
    schedule: tuple[ScheduledBatch, ...],
) -> int:
    return sum(scheduled_batch_flops(model, batch) for batch in schedule)


def _initial_state(
    model: RecurrentReasoner,
    batch_size: int,
    batch: ScheduledBatch,
    standard_deviation: float,
):
    if model.arm_id == "fixed-depth-v1" or batch.use_zero_initial_state:
        return mx.zeros((batch_size, model.config.hidden_dimensions))
    key = mx.random.key(batch.initial_state_seed)
    return mx.random.normal(
        (batch_size, model.config.hidden_dimensions), key=key
    ) * standard_deviation


def _loss_function(
    model: RecurrentReasoner,
    tokens: mx.array,
    targets: mx.array,
    target_mask: mx.array,
    macro_steps: int,
    micro_steps: int,
    initial_state: mx.array,
    intermediate_weight: float,
    halt_targets: mx.array,
    halt_weight: float,
) -> mx.array:
    outputs = model(
        tokens,
        macro_steps=macro_steps,
        micro_steps=micro_steps,
        initial_state=initial_state,
    )
    answer_loss = token_loss(
        outputs,
        targets,
        target_mask,
        intermediate_weight=intermediate_weight,
    )
    halt_loss = nn.losses.binary_cross_entropy(
        outputs["halt_logits"],
        halt_targets,
        with_logits=True,
        reduction="mean",
    )
    return answer_loss + halt_weight * halt_loss


def _halt_targets(
    model: RecurrentReasoner,
    tasks: list[RecurrentTask],
    *,
    macro_steps: int,
    micro_steps: int,
) -> mx.array:
    if model.arm_id == "fixed-depth-v1":
        state_count = model.config.fixed_reasoning_blocks
    elif model.arm_id == "flat-recurrent-v1":
        state_count = macro_steps
    else:
        state_count = macro_steps * micro_steps
    return mx.array(
        [
            [
                1.0 if position >= task.required_reasoning_steps else 0.0
                for position in range(1, state_count + 1)
            ]
            for task in tasks
        ],
        dtype=mx.float32,
    )


def train_schedule_segment(
    model: RecurrentReasoner,
    codec: ByteCodec,
    tasks: tuple[RecurrentTask, ...],
    schedule: tuple[ScheduledBatch, ...],
    config: OptimizerConfig,
    *,
    optimizer,
    start_update: int,
    stop_update: int,
) -> dict[str, Any]:
    if not 0 <= start_update <= stop_update <= len(schedule):
        raise RecurrentTrainingError("training segment is outside the frozen schedule")
    by_id = {task.task_id: task for task in tasks}
    loss_and_grad = nn.value_and_grad(model, _loss_function)
    losses = []
    gradient_norms = []
    examples_seen = 0
    realized_training_flops = 0
    for expected_update, batch in enumerate(
        schedule[start_update:stop_update], start=start_update
    ):
        if batch.update != expected_update:
            raise RecurrentTrainingError("schedule update identities are not contiguous")
        batch_tasks = [by_id[task_id] for task_id in batch.task_ids]
        tokens, targets, target_mask = batch_arrays(codec, batch_tasks)
        initial_state = _initial_state(
            model,
            len(batch_tasks),
            batch,
            config.initial_state_standard_deviation,
        )
        macro_steps = 4 if model.arm_id == "fixed-depth-v1" else batch.macro_steps
        halt_targets = _halt_targets(
            model,
            batch_tasks,
            macro_steps=macro_steps,
            micro_steps=batch.micro_steps,
        )
        loss, gradients = loss_and_grad(
            model,
            tokens,
            targets,
            target_mask,
            macro_steps,
            batch.micro_steps,
            initial_state,
            config.intermediate_state_loss_weight,
            halt_targets,
            config.halt_binary_cross_entropy_weight,
        )
        gradients, gradient_norm = optim.clip_grad_norm(
            gradients, config.gradient_clip_norm
        )
        optimizer.update(model, gradients)
        # Canonicalize moment storage after every update. A safetensor-restored
        # float32 moment can be numerically identical yet have a different
        # internal layout; allowing the next fused update to depend on that
        # layout breaks bitwise checkpoint/resume equivalence.
        optimizer.state = tree_unflatten(
            [
                (name, value + mx.zeros_like(value))
                for name, value in tree_flatten(optimizer.state)
            ]
        )
        model.load_weights(
            [
                (name, value + mx.zeros_like(value))
                for name, value in tree_flatten(model.trainable_parameters())
            ],
            strict=True,
        )
        mx.eval(loss, gradient_norm, model.parameters(), optimizer.state)
        loss_value = float(loss)
        gradient_norm_value = float(gradient_norm)
        if not math.isfinite(loss_value) or not math.isfinite(gradient_norm_value):
            raise RecurrentTrainingError("training produced a non-finite value")
        losses.append(loss_value)
        gradient_norms.append(gradient_norm_value)
        examples_seen += len(batch_tasks)
        realized_training_flops += scheduled_batch_flops(model, batch)
    if not losses or any(not (loss >= 0.0) for loss in losses):
        raise RecurrentTrainingError("training produced an invalid loss sequence")
    return {
        "arm_id": model.arm_id,
        "start_update": start_update,
        "stop_update": stop_update,
        "updates": stop_update - start_update,
        "examples_seen": examples_seen,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "maximum_gradient_norm": max(gradient_norms),
        "losses": losses,
        "gradient_norms": gradient_norms,
        "realized_training_flops": realized_training_flops,
        "schedule_hash": schedule_hash(schedule),
        "parameter_count": parameter_count(model),
    }


def train_updates(
    model: RecurrentReasoner,
    codec: ByteCodec,
    tasks: tuple[RecurrentTask, ...],
    schedule: tuple[ScheduledBatch, ...],
    config: OptimizerConfig,
) -> dict[str, Any]:
    optimizer = create_optimizer(config, total_updates=len(schedule))
    return train_schedule_segment(
        model,
        codec,
        tasks,
        schedule,
        config,
        optimizer=optimizer,
        start_update=0,
        stop_update=len(schedule),
    )


def evaluate_exact(
    model: RecurrentReasoner,
    codec: ByteCodec,
    tasks: list[RecurrentTask],
    *,
    macro_steps: int,
    micro_steps: int,
) -> dict[str, Any]:
    tokens, _, _ = batch_arrays(codec, tasks)
    outputs = model(tokens, macro_steps=macro_steps, micro_steps=micro_steps)
    mx.eval(outputs["logits"])
    predicted = decode_batch(codec, outputs["logits"])
    exact = [prediction == task.expected for prediction, task in zip(predicted, tasks)]
    return {
        "examples": len(tasks),
        "exact": sum(exact),
        "accuracy": sum(exact) / len(exact) if exact else 0.0,
        "macro_steps": macro_steps,
        "micro_steps": micro_steps,
    }


def compute_record_for_batch(
    model: RecurrentReasoner,
    tasks: list[RecurrentTask],
    *,
    macro_steps: int,
    micro_steps: int,
    training_examples_seen: int,
    optimizer_updates: int,
) -> ComputeRecord:
    flops = model.estimated_inference_flops(
        input_tokens=model.config.maximum_input_tokens,
        macro_steps=macro_steps,
        micro_steps=micro_steps,
    )
    record = ComputeRecord(
        arm_id=model.arm_id,
        task_id=f"batch:{sha256_json([task.task_id for task in tasks])[:16]}",
        parameter_count=parameter_count(model),
        inference_flops=flops,
        macro_steps=(4 if model.arm_id == "fixed-depth-v1" else macro_steps),
        micro_steps=(
            4
            if model.arm_id == "fixed-depth-v1"
            else macro_steps
            if model.arm_id == "flat-recurrent-v1"
            else macro_steps * micro_steps
        ),
        retrieval_calls=0,
        tool_calls=0,
        training_examples_seen=training_examples_seen,
        optimizer_updates=optimizer_updates,
    )
    budget = ComputeBudget(
        maximum_inference_flops=model.config.maximum_inference_flops,
        maximum_macro_steps=12,
        maximum_micro_steps_per_macro=4,
        maximum_total_micro_steps=48,
        maximum_retrieval_calls=4,
        maximum_tool_calls=4,
    )
    record.validate(budget)
    return record


def validate_shared_schedule(
    reports: list[dict[str, Any]],
) -> None:
    if len(reports) != len(ARM_IDS):
        raise RecurrentTrainingError("one report per comparison arm is required")
    if len({report["schedule_hash"] for report in reports}) != 1:
        raise RecurrentTrainingError("comparison arms did not use the same batches/depths")
    if len({report["examples_seen"] for report in reports}) != 1:
        raise RecurrentTrainingError("comparison arms saw different example counts")
