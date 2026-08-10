"""Registered run accounting and statistics for Milestone 3.

This module stays standard-library-only.  The optional MLX runner imports it,
but the memory kernel and its tests do not acquire a neural dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from .recurrent_model_constants import RECURRENT_ARM_IDS

RUN_ACCOUNTING_SCHEMA_VERSION = 1
RUN_COUNTING_METHOD = "wamrx-analytical-flops-v2"
RUN_PHASE_STATUSES = frozenset({"NOT_RUN", "INCOMPLETE", "INVALID", "COMPLETE"})
TERMINAL_DECISIONS = frozenset(
    {
        "COMPLETE_RETAIN_FIXED",
        "COMPLETE_ADOPT_FLAT",
        "COMPLETE_ADOPT_HIERARCHY",
        "COMPLETE_SPECIALIST_ONLY",
        "COMPLETE_COMPUTE_SCALING_ONLY",
    }
)


class RecurrentRunError(ValueError):
    pass


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecurrentRunError(f"{field} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class RunComputeRecord:
    run_id: str
    arm_id: str
    seed: int
    status: str
    parameter_count: int
    schedule_hash: str
    estimated_training_flops: int
    realized_training_flops: int
    estimated_inference_flops: int
    realized_inference_flops: int
    training_examples_planned: int
    training_examples_seen: int
    optimizer_updates_planned: int
    optimizer_updates_completed: int
    evaluation_examples_planned: int
    evaluation_examples_completed: int
    retrieval_calls: int = 0
    tool_calls: int = 0
    counting_method: str = RUN_COUNTING_METHOD
    schema_version: int = RUN_ACCOUNTING_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != RUN_ACCOUNTING_SCHEMA_VERSION:
            raise RecurrentRunError("unsupported run-accounting schema")
        if not self.run_id:
            raise RecurrentRunError("run accounting requires a run ID")
        if self.arm_id not in RECURRENT_ARM_IDS:
            raise RecurrentRunError(f"unknown comparison arm {self.arm_id!r}")
        if self.status not in RUN_PHASE_STATUSES:
            raise RecurrentRunError(f"invalid run phase status {self.status!r}")
        if self.counting_method != RUN_COUNTING_METHOD:
            raise RecurrentRunError("unregistered FLOP-counting method")
        if len(self.schedule_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.schedule_hash
        ):
            raise RecurrentRunError("schedule hash must be lowercase SHA-256")
        fields = (
            "seed",
            "parameter_count",
            "estimated_training_flops",
            "realized_training_flops",
            "estimated_inference_flops",
            "realized_inference_flops",
            "training_examples_planned",
            "training_examples_seen",
            "optimizer_updates_planned",
            "optimizer_updates_completed",
            "evaluation_examples_planned",
            "evaluation_examples_completed",
            "retrieval_calls",
            "tool_calls",
        )
        for field in fields:
            _nonnegative_integer(getattr(self, field), field)
        if self.parameter_count == 0:
            raise RecurrentRunError("parameter count must be positive")
        if self.realized_training_flops > self.estimated_training_flops:
            raise RecurrentRunError("realized training FLOPs exceed the frozen estimate")
        if self.training_examples_seen > self.training_examples_planned:
            raise RecurrentRunError("training examples exceed the frozen schedule")
        if self.optimizer_updates_completed > self.optimizer_updates_planned:
            raise RecurrentRunError("optimizer updates exceed the frozen schedule")
        if self.evaluation_examples_completed > self.evaluation_examples_planned:
            raise RecurrentRunError("evaluation examples exceed the frozen plan")
        if self.status == "COMPLETE" and (
            self.optimizer_updates_completed != self.optimizer_updates_planned
            or self.training_examples_seen != self.training_examples_planned
            or self.evaluation_examples_completed != self.evaluation_examples_planned
        ):
            raise RecurrentRunError("a complete record has incomplete work counters")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "arm_id": self.arm_id,
            "seed": self.seed,
            "status": self.status,
            "counting_method": self.counting_method,
            "parameter_count": self.parameter_count,
            "schedule_hash": self.schedule_hash,
            "estimated_training_flops": self.estimated_training_flops,
            "realized_training_flops": self.realized_training_flops,
            "estimated_inference_flops": self.estimated_inference_flops,
            "realized_inference_flops": self.realized_inference_flops,
            "training_examples_planned": self.training_examples_planned,
            "training_examples_seen": self.training_examples_seen,
            "optimizer_updates_planned": self.optimizer_updates_planned,
            "optimizer_updates_completed": self.optimizer_updates_completed,
            "evaluation_examples_planned": self.evaluation_examples_planned,
            "evaluation_examples_completed": self.evaluation_examples_completed,
            "retrieval_calls": self.retrieval_calls,
            "tool_calls": self.tool_calls,
        }


def _continued_fraction_beta(a: float, b: float, x: float) -> float:
    maximum_iterations = 300
    epsilon = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        even = 2 * iteration
        numerator = iteration * (b - iteration) * x / (
            (qam + even) * (a + even)
        )
        d = 1.0 + numerator * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + numerator / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c
        numerator = -(a + iteration) * (qab + iteration) * x / (
            (a + even) * (qap + even)
        )
        d = 1.0 + numerator * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + numerator / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result
    raise RecurrentRunError("incomplete-beta continued fraction did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not 0.0 <= x <= 1.0:
        raise RecurrentRunError("incomplete-beta input must be in [0, 1]")
    if x in {0.0, 1.0}:
        return x
    factor = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _continued_fraction_beta(a, b, x) / a
    return 1.0 - factor * _continued_fraction_beta(b, a, 1.0 - x) / b


def student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom < 1:
        raise RecurrentRunError("Student-t degrees of freedom must be positive")
    if math.isnan(value):
        raise RecurrentRunError("Student-t input cannot be NaN")
    if math.isinf(value):
        return 1.0 if value > 0 else 0.0
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * _regularized_incomplete_beta(
        degrees_of_freedom / 2.0, 0.5, x
    )
    return 1.0 - tail if value >= 0.0 else tail


def student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    if not 0.0 < probability < 1.0:
        raise RecurrentRunError("Student-t probability must be in (0, 1)")
    low, high = -64.0, 64.0
    for _ in range(160):
        midpoint = (low + high) / 2.0
        if student_t_cdf(midpoint, degrees_of_freedom) < probability:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def paired_student_t(
    deltas: Iterable[float],
    *,
    null_margin: float,
    alpha: float,
) -> dict[str, Any]:
    values = tuple(float(value) for value in deltas)
    if len(values) < 2:
        raise RecurrentRunError("paired Student-t requires at least two deltas")
    if not 0.0 < alpha < 1.0:
        raise RecurrentRunError("alpha must be in (0, 1)")
    if any(not math.isfinite(value) for value in values):
        raise RecurrentRunError("paired deltas must be finite")
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    standard_deviation = math.sqrt(variance)
    standard_error = standard_deviation / math.sqrt(count)
    if standard_error == 0.0:
        statistic = None
        p_value = 0.0 if mean > null_margin else 1.0
        lower_bound = mean
    else:
        statistic = (mean - null_margin) / standard_error
        p_value = 1.0 - student_t_cdf(statistic, count - 1)
        critical = student_t_quantile(1.0 - alpha, count - 1)
        lower_bound = mean - critical * standard_error
    return {
        "pairs": count,
        "mean_delta": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "null_margin": null_margin,
        "alpha": alpha,
        "t_statistic": statistic,
        "one_sided_p_value": p_value,
        "one_sided_lower_bound": lower_bound,
        "passes": lower_bound >= null_margin,
    }


def holm_adjusted_tests(
    hypotheses: dict[str, tuple[Iterable[float], float]],
    *,
    family_alpha: float = 0.05,
) -> dict[str, Any]:
    if not hypotheses:
        raise RecurrentRunError("Holm correction requires at least one hypothesis")
    nominal = {
        name: paired_student_t(values, null_margin=margin, alpha=family_alpha)
        for name, (values, margin) in hypotheses.items()
    }
    ordered = sorted(
        nominal,
        key=lambda name: (nominal[name]["one_sided_p_value"], name),
    )
    active = True
    results: dict[str, Any] = {}
    count = len(ordered)
    for rank, name in enumerate(ordered, start=1):
        adjusted_alpha = family_alpha / (count - rank + 1)
        values, margin = hypotheses[name]
        test = paired_student_t(values, null_margin=margin, alpha=adjusted_alpha)
        rejected = active and test["one_sided_p_value"] <= adjusted_alpha
        if not rejected:
            active = False
        results[name] = {
            **test,
            "holm_rank": rank,
            "holm_alpha": adjusted_alpha,
            "holm_rejects_null": rejected,
        }
    return {
        "procedure": "Holm step-down one-sided paired Student-t",
        "family_alpha": family_alpha,
        "ordered_hypotheses": ordered,
        "hypotheses": results,
    }


def resolve_terminal_status(
    *,
    invalid_reasons: Iterable[str],
    incomplete_reasons: Iterable[str],
    decision: str | None,
) -> str:
    invalid = tuple(invalid_reasons)
    incomplete = tuple(incomplete_reasons)
    if invalid:
        return "INVALID"
    if incomplete:
        return "INCOMPLETE"
    if decision is None:
        return "NOT_RUN"
    if decision not in TERMINAL_DECISIONS:
        raise RecurrentRunError(f"unknown terminal decision {decision!r}")
    return decision
