"""Small MLX reasoners for the frozen Milestone 3 three-arm comparison.

This module is intentionally isolated from :mod:`wamrx.__init__` so the
Milestone 1/2 standard-library kernel remains importable without MLX.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .canonical import canonical_json

MODEL_IMPLEMENTATION_VERSION = "wamrx-recurrent-mlx-v1"
ARM_IDS = (
    "fixed-depth-v1",
    "flat-recurrent-v1",
    "hierarchical-recurrent-v1",
)


class RecurrentModelError(ValueError):
    pass


@dataclass(frozen=True)
class RecurrentModelConfig:
    vocabulary_size: int
    hidden_dimensions: int
    feedforward_dimensions: int
    maximum_input_tokens: int
    maximum_output_tokens: int
    fixed_reasoning_blocks: int
    flat_core_blocks: int
    hierarchical_high_blocks: int
    hierarchical_low_blocks: int
    parameter_targets: dict[str, int]
    maximum_inference_flops: int

    @classmethod
    def load(cls, path: str | Path) -> "RecurrentModelConfig":
        value = json.loads(Path(path).read_text())
        if value.get("training_config_schema_version") != 1:
            raise RecurrentModelError("unsupported recurrent training config")
        model = value["model"]
        config = cls(
            vocabulary_size=int(model["vocabulary_size"]),
            hidden_dimensions=int(model["hidden_dimensions"]),
            feedforward_dimensions=int(model["feedforward_dimensions"]),
            maximum_input_tokens=int(model["maximum_input_tokens"]),
            maximum_output_tokens=int(model["maximum_output_tokens"]),
            fixed_reasoning_blocks=int(model["fixed_reasoning_blocks"]),
            flat_core_blocks=int(model["flat_core_blocks"]),
            hierarchical_high_blocks=int(model["hierarchical_high_blocks"]),
            hierarchical_low_blocks=int(model["hierarchical_low_blocks"]),
            parameter_targets={
                str(key): int(count)
                for key, count in model["parameter_targets"].items()
            },
            maximum_inference_flops=int(
                value["evaluation"]["maximum_inference_flops_per_example"]
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if set(self.parameter_targets) != set(ARM_IDS):
            raise RecurrentModelError("training config must target all three arms")
        if self.vocabulary_size != ByteCodec.VOCABULARY_SIZE:
            raise RecurrentModelError("training vocabulary does not match byte codec")
        if self.fixed_reasoning_blocks != self.flat_core_blocks:
            raise RecurrentModelError("fixed and flat cores must contain equal block counts")
        if (
            self.hierarchical_high_blocks + self.hierarchical_low_blocks
            != self.flat_core_blocks
        ):
            raise RecurrentModelError(
                "hierarchical high/low block count must equal the flat core"
            )
        if min(
            self.hidden_dimensions,
            self.feedforward_dimensions,
            self.maximum_input_tokens,
            self.maximum_output_tokens,
            self.maximum_inference_flops,
        ) <= 0:
            raise RecurrentModelError("model dimensions and FLOP budget must be positive")


class ByteCodec:
    """Canonical JSON to a fixed byte vocabulary shared by every arm."""

    BOS = 256
    EOS = 257
    PAD = 258
    UNK = 259
    VOCABULARY_SIZE = 260

    def __init__(self, *, maximum_input_tokens: int, maximum_output_tokens: int):
        self.maximum_input_tokens = maximum_input_tokens
        self.maximum_output_tokens = maximum_output_tokens

    @staticmethod
    def _encode_bytes(value: Any) -> list[int]:
        return list(canonical_json(value).encode("utf-8"))

    @staticmethod
    def _fixed(tokens: list[int], length: int) -> tuple[list[int], list[float]]:
        if len(tokens) > length:
            raise RecurrentModelError(
                f"encoded value requires {len(tokens)} tokens but limit is {length}"
            )
        mask = [1.0] * len(tokens)
        return (
            [*tokens, *([ByteCodec.PAD] * (length - len(tokens)))],
            [*mask, *([0.0] * (length - len(tokens)))],
        )

    def encode_problem(self, problem: Any) -> tuple[list[int], list[float]]:
        tokens = [self.BOS, *self._encode_bytes(problem), self.EOS]
        return self._fixed(tokens, self.maximum_input_tokens)

    def encode_answer(self, answer: Any) -> tuple[list[int], list[float]]:
        tokens = [*self._encode_bytes(answer), self.EOS]
        return self._fixed(tokens, self.maximum_output_tokens)

    def decode_answer(self, tokens: list[int]) -> Any:
        output = []
        for token in tokens:
            if token == self.EOS:
                break
            if 0 <= token <= 255:
                output.append(token)
        try:
            return json.loads(bytes(output).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None


class ResidualMLP(nn.Module):
    def __init__(self, dimensions: int, feedforward_dimensions: int):
        super().__init__()
        self.norm = nn.LayerNorm(dimensions)
        self.up = nn.Linear(dimensions, feedforward_dimensions)
        self.down = nn.Linear(feedforward_dimensions, dimensions)

    def __call__(self, state: mx.array) -> mx.array:
        hidden = nn.gelu(self.up(self.norm(state)))
        return state + self.down(hidden)


class CapacityMatch(nn.Module):
    """A used low-rank residual plus partial-width biases with an exact budget."""

    def __init__(self, dimensions: int, parameter_budget: int):
        super().__init__()
        if parameter_budget < 0:
            raise RecurrentModelError("base model already exceeds parameter target")
        self.dimensions = dimensions
        self.rank = parameter_budget // (2 * dimensions)
        used = 2 * dimensions * self.rank
        if self.rank:
            self.up = nn.Linear(dimensions, self.rank, bias=False)
            self.down = nn.Linear(self.rank, dimensions, bias=False)
        self.bias_chunks = []
        remaining = parameter_budget - used
        while remaining:
            size = min(dimensions, remaining)
            self.bias_chunks.append(mx.zeros((size,)))
            remaining -= size

    def __call__(self, state: mx.array) -> mx.array:
        if self.rank:
            state = state + self.down(nn.gelu(self.up(state)))
        for chunk in self.bias_chunks:
            padding = self.dimensions - chunk.shape[0]
            bias = mx.pad(chunk, ((0, padding),)) if padding else chunk
            state = state + bias
        return state

    def estimated_flops(self) -> int:
        return 4 * self.dimensions * self.rank


class SharedPrelude(nn.Module):
    def __init__(self, config: RecurrentModelConfig):
        super().__init__()
        dimensions = config.hidden_dimensions
        self.token_embedding = nn.Embedding(config.vocabulary_size, dimensions)
        self.position_embedding = nn.Embedding(
            config.maximum_input_tokens, dimensions
        )
        self.norm = nn.LayerNorm(dimensions)
        self.projection = nn.Linear(dimensions, dimensions)

    def __call__(self, tokens: mx.array) -> mx.array:
        length = tokens.shape[1]
        positions = mx.arange(length)
        embedded = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = (tokens != ByteCodec.PAD)[..., None]
        denominator = mx.maximum(mx.sum(mask, axis=1), 1)
        pooled = mx.sum(embedded * mask, axis=1) / denominator
        return self.projection(self.norm(pooled))


class SharedCoda(nn.Module):
    def __init__(self, config: RecurrentModelConfig):
        super().__init__()
        dimensions = config.hidden_dimensions
        self.maximum_output_tokens = config.maximum_output_tokens
        self.position_embedding = nn.Embedding(
            config.maximum_output_tokens, dimensions
        )
        self.norm = nn.LayerNorm(dimensions)
        self.output = nn.Linear(dimensions, config.vocabulary_size)

    def __call__(self, state: mx.array) -> mx.array:
        positions = self.position_embedding(mx.arange(self.maximum_output_tokens))
        hidden = state[:, None, :] + positions[None, :, :]
        return self.output(self.norm(hidden))


def parameter_count(module: nn.Module) -> int:
    return sum(array.size for _, array in tree_flatten(module.trainable_parameters()))


class RecurrentReasoner(nn.Module):
    """One of the three frozen arms with shared prelude/coda shapes."""

    def __init__(self, arm_id: str, config: RecurrentModelConfig):
        super().__init__()
        if arm_id not in ARM_IDS:
            raise RecurrentModelError(f"unknown recurrent arm {arm_id!r}")
        self.arm_id = arm_id
        self.config = config
        self.prelude = SharedPrelude(config)
        self.blocks = [
            ResidualMLP(
                config.hidden_dimensions,
                config.feedforward_dimensions,
            )
            for _ in range(config.flat_core_blocks)
        ]
        self.coda = SharedCoda(config)
        self.halt_head = nn.Linear(config.hidden_dimensions, 1)
        base_count = parameter_count(self)
        self.capacity_match = CapacityMatch(
            config.hidden_dimensions,
            config.parameter_targets[arm_id] - base_count,
        )
        actual_count = parameter_count(self)
        if actual_count != config.parameter_targets[arm_id]:
            raise RecurrentModelError(
                f"{arm_id} parameter target mismatch: "
                f"expected {config.parameter_targets[arm_id]}, got {actual_count}"
            )

    def _inject(self, state: mx.array, context: mx.array) -> mx.array:
        return self.capacity_match(state + context)

    def encode(self, tokens: mx.array) -> mx.array:
        return self.prelude(tokens)

    def state_outputs(self, state: mx.array) -> tuple[mx.array, mx.array]:
        return self.coda(state), self.halt_head(state)

    def flat_step(self, context: mx.array, state: mx.array) -> mx.array:
        state = self._inject(state, context)
        for block in self.blocks:
            state = block(state)
        return state

    def hierarchical_step(
        self,
        context: mx.array,
        high: mx.array,
        low: mx.array,
        micro_steps: int,
    ) -> tuple[mx.array, mx.array, list[mx.array]]:
        high_blocks = self.blocks[: self.config.hierarchical_high_blocks]
        low_block = self.blocks[-1]
        high = self._inject(high, context)
        for block in high_blocks:
            high = block(high)
        states = []
        for _ in range(micro_steps):
            low = low_block(self._inject(low + high, context))
            states.append((high + low) * 0.5)
        return high, low, states

    def _fixed(self, context: mx.array) -> tuple[mx.array, list[mx.array]]:
        state = mx.zeros_like(context)
        states = []
        for block in self.blocks:
            state = block(self._inject(state, context))
            states.append(state)
        return state, states

    def _flat(
        self,
        context: mx.array,
        macro_steps: int,
        initial_state: mx.array | None,
    ) -> tuple[mx.array, list[mx.array]]:
        state = mx.zeros_like(context) if initial_state is None else initial_state
        states = []
        for _ in range(macro_steps):
            state = self.flat_step(context, state)
            states.append(state)
        return state, states

    def _hierarchical(
        self,
        context: mx.array,
        macro_steps: int,
        micro_steps: int,
        initial_state: mx.array | None,
    ) -> tuple[mx.array, list[mx.array]]:
        high = mx.zeros_like(context) if initial_state is None else initial_state
        low = mx.zeros_like(context)
        states = []
        for _ in range(macro_steps):
            high, low, micro_states = self.hierarchical_step(
                context, high, low, micro_steps
            )
            states.extend(micro_states)
        return (high + low) * 0.5, states

    def __call__(
        self,
        tokens: mx.array,
        *,
        macro_steps: int = 4,
        micro_steps: int = 1,
        initial_state: mx.array | None = None,
    ) -> dict[str, Any]:
        if macro_steps < 1 or micro_steps < 1:
            raise RecurrentModelError("macro and micro step counts must be positive")
        context = self.encode(tokens)
        if self.arm_id == "fixed-depth-v1":
            final_state, states = self._fixed(context)
        elif self.arm_id == "flat-recurrent-v1":
            final_state, states = self._flat(context, macro_steps, initial_state)
        else:
            final_state, states = self._hierarchical(
                context,
                macro_steps,
                micro_steps,
                initial_state,
            )
        return {
            "logits": self.coda(final_state),
            "halt_logits": mx.concatenate(
                [self.halt_head(state) for state in states], axis=-1
            ),
            "intermediate_logits": [self.coda(state) for state in states],
            "final_state": final_state,
            "states": states,
        }

    def estimated_inference_flops(
        self,
        *,
        input_tokens: int,
        macro_steps: int,
        micro_steps: int,
    ) -> int:
        d = self.config.hidden_dimensions
        ff = self.config.feedforward_dimensions
        embedding_and_pool = input_tokens * d * 2
        projection = 2 * d * d
        block = 4 * d * ff
        injection = self.capacity_match.estimated_flops() + d
        if self.arm_id == "fixed-depth-v1":
            block_calls = self.config.fixed_reasoning_blocks
            injection_calls = block_calls
        elif self.arm_id == "flat-recurrent-v1":
            block_calls = self.config.flat_core_blocks * macro_steps
            injection_calls = macro_steps
        else:
            block_calls = macro_steps * (
                self.config.hierarchical_high_blocks
                + self.config.hierarchical_low_blocks * micro_steps
            )
            injection_calls = macro_steps * (1 + micro_steps)
        coda = (
            2
            * self.config.maximum_output_tokens
            * d
            * self.config.vocabulary_size
        )
        halt = 2 * d * max(1, macro_steps * micro_steps)
        return int(
            embedding_and_pool
            + projection
            + block * block_calls
            + injection * injection_calls
            + coda
            + halt
        )


def token_loss(
    outputs: dict[str, Any],
    targets: mx.array,
    target_mask: mx.array,
    *,
    intermediate_weight: float,
) -> mx.array:
    def sequence_loss(logits: mx.array) -> mx.array:
        losses = nn.losses.cross_entropy(logits, targets, reduction="none")
        return mx.sum(losses * target_mask) / mx.maximum(mx.sum(target_mask), 1.0)

    final = sequence_loss(outputs["logits"])
    intermediates = outputs["intermediate_logits"][:-1]
    if not intermediates or intermediate_weight == 0.0:
        return final
    intermediate = mx.mean(mx.stack([sequence_loss(logits) for logits in intermediates]))
    return final + intermediate_weight * intermediate


def batch_arrays(
    codec: ByteCodec,
    tasks: list[Any],
) -> tuple[mx.array, mx.array, mx.array]:
    problems = []
    targets = []
    target_masks = []
    for task in tasks:
        problem, _ = codec.encode_problem(task.problem)
        target, target_mask = codec.encode_answer(task.expected)
        problems.append(problem)
        targets.append(target)
        target_masks.append(target_mask)
    return (
        mx.array(problems, dtype=mx.int32),
        mx.array(targets, dtype=mx.int32),
        mx.array(target_masks, dtype=mx.float32),
    )


def decode_batch(codec: ByteCodec, logits: mx.array) -> list[Any]:
    token_ids = mx.argmax(logits, axis=-1).tolist()
    return [codec.decode_answer(tokens) for tokens in token_ids]
