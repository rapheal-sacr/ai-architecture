"""Fail-closed interface, trace, compute, and halting contracts for Milestone 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .artifacts import SupportManifest
from .canonical import sha256_json

RECURRENT_INTERFACE_VERSION = "wamrx-recurrent-interface-v1"
TRACE_SCHEMA_VERSION = 1
COMPUTE_SCHEMA_VERSION = 1
HALT_REASONS = frozenset(
    {"continue", "resolved", "resolved_at_budget", "unresolved_budget"}
)


class RecurrentContractError(ValueError):
    pass


def _nonempty_strings(values: Iterable[str], field: str) -> tuple[str, ...]:
    items = tuple(values)
    if any(not isinstance(item, str) or not item for item in items):
        raise RecurrentContractError(f"{field} must contain non-empty strings")
    return items


@dataclass(frozen=True)
class ComputeBudget:
    maximum_inference_flops: int
    maximum_macro_steps: int
    maximum_micro_steps_per_macro: int
    maximum_total_micro_steps: int
    maximum_retrieval_calls: int
    maximum_tool_calls: int

    def validate(self) -> None:
        values = self.to_dict()
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()):
            raise RecurrentContractError("compute-budget fields must be nonnegative integers")
        if self.maximum_inference_flops == 0 or self.maximum_macro_steps == 0:
            raise RecurrentContractError("FLOP and macro-step budgets must be positive")
        if self.maximum_micro_steps_per_macro == 0 or self.maximum_total_micro_steps == 0:
            raise RecurrentContractError("micro-step budgets must be positive")
        if (
            self.maximum_total_micro_steps
            > self.maximum_macro_steps * self.maximum_micro_steps_per_macro
        ):
            raise RecurrentContractError(
                "total micro-step budget exceeds macro-times-micro capacity"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "maximum_inference_flops": self.maximum_inference_flops,
            "maximum_macro_steps": self.maximum_macro_steps,
            "maximum_micro_steps_per_macro": self.maximum_micro_steps_per_macro,
            "maximum_total_micro_steps": self.maximum_total_micro_steps,
            "maximum_retrieval_calls": self.maximum_retrieval_calls,
            "maximum_tool_calls": self.maximum_tool_calls,
        }


@dataclass(frozen=True)
class ComputeRecord:
    arm_id: str
    task_id: str
    parameter_count: int
    inference_flops: int
    macro_steps: int
    micro_steps: int
    retrieval_calls: int
    tool_calls: int
    training_examples_seen: int
    optimizer_updates: int
    schema_version: int = COMPUTE_SCHEMA_VERSION

    def validate(self, budget: ComputeBudget) -> None:
        budget.validate()
        if self.schema_version != COMPUTE_SCHEMA_VERSION:
            raise RecurrentContractError("unsupported compute-accounting schema")
        if not self.arm_id or not self.task_id:
            raise RecurrentContractError("compute records require arm and task IDs")
        values = {
            "parameter_count": self.parameter_count,
            "inference_flops": self.inference_flops,
            "macro_steps": self.macro_steps,
            "micro_steps": self.micro_steps,
            "retrieval_calls": self.retrieval_calls,
            "tool_calls": self.tool_calls,
            "training_examples_seen": self.training_examples_seen,
            "optimizer_updates": self.optimizer_updates,
        }
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()):
            raise RecurrentContractError("compute-accounting values must be nonnegative integers")
        limits = {
            "inference_flops": budget.maximum_inference_flops,
            "macro_steps": budget.maximum_macro_steps,
            "micro_steps": budget.maximum_total_micro_steps,
            "retrieval_calls": budget.maximum_retrieval_calls,
            "tool_calls": budget.maximum_tool_calls,
        }
        exceeded = {
            key: {"actual": values[key], "maximum": maximum}
            for key, maximum in limits.items()
            if values[key] > maximum
        }
        if exceeded:
            raise RecurrentContractError(f"compute budget exceeded: {exceeded}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "arm_id": self.arm_id,
            "task_id": self.task_id,
            "parameter_count": self.parameter_count,
            "inference_flops": self.inference_flops,
            "macro_steps": self.macro_steps,
            "micro_steps": self.micro_steps,
            "retrieval_calls": self.retrieval_calls,
            "tool_calls": self.tool_calls,
            "training_examples_seen": self.training_examples_seen,
            "optimizer_updates": self.optimizer_updates,
        }


@dataclass(frozen=True)
class EvidenceViewReference:
    view_type: str
    artifact_id: str
    content_hash: str
    ledger_frontier_sequence: int
    ledger_frontier_hash: str
    ontology_version: str
    component_versions: dict[str, str]
    support_event_ids: tuple[str, ...]
    protected_regions: tuple[str, ...]
    compatible_with_runtime: bool = True
    support_usable: bool = True

    def validate(self) -> None:
        required = {
            "view_type": self.view_type,
            "artifact_id": self.artifact_id,
            "content_hash": self.content_hash,
            "ledger_frontier_hash": self.ledger_frontier_hash,
            "ontology_version": self.ontology_version,
        }
        if any(not isinstance(value, str) or not value for value in required.values()):
            raise RecurrentContractError("evidence reference stamp is incomplete")
        if self.ledger_frontier_sequence < 0:
            raise RecurrentContractError("evidence frontier sequence cannot be negative")
        if not self.component_versions or any(
            not key or not value for key, value in self.component_versions.items()
        ):
            raise RecurrentContractError("evidence component versions are incomplete")
        _nonempty_strings(self.support_event_ids, "support_event_ids")
        _nonempty_strings(self.protected_regions, "protected_regions")
        if not self.compatible_with_runtime:
            raise RecurrentContractError(
                f"evidence artifact {self.artifact_id!r} is runtime-incompatible"
            )
        if not self.support_usable:
            raise RecurrentContractError(
                f"evidence artifact {self.artifact_id!r} has tombstoned support"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_type": self.view_type,
            "artifact_id": self.artifact_id,
            "content_hash": self.content_hash,
            "ledger_frontier_sequence": self.ledger_frontier_sequence,
            "ledger_frontier_hash": self.ledger_frontier_hash,
            "ontology_version": self.ontology_version,
            "component_versions": dict(sorted(self.component_versions.items())),
            "support_event_ids": list(self.support_event_ids),
            "protected_regions": list(self.protected_regions),
            "compatible_with_runtime": self.compatible_with_runtime,
            "support_usable": self.support_usable,
        }


@dataclass(frozen=True)
class EvidenceBundle:
    bundle_id: str
    problem_hash: str
    views: tuple[EvidenceViewReference, ...]
    active_ontology_version: str
    active_runtime_id: str

    def validate(self, *, required_regions: tuple[str, ...] = ()) -> None:
        if not self.bundle_id or not self.problem_hash or not self.active_runtime_id:
            raise RecurrentContractError("evidence bundle identity is incomplete")
        if not self.active_ontology_version or not self.views:
            raise RecurrentContractError("evidence bundle ontology and views are required")
        artifact_ids = [view.artifact_id for view in self.views]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise RecurrentContractError("evidence bundle repeats an artifact ID")
        for view in self.views:
            view.validate()
            if view.ontology_version != self.active_ontology_version:
                raise RecurrentContractError(
                    f"evidence artifact {view.artifact_id!r} has a stale ontology"
                )
        covered = {region for view in self.views for region in view.protected_regions}
        missing = sorted(set(required_regions) - covered)
        if missing:
            raise RecurrentContractError(
                f"evidence bundle omits protected regions: {missing}"
            )

    @property
    def bundle_hash(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "problem_hash": self.problem_hash,
            "views": [view.to_dict() for view in self.views],
            "active_ontology_version": self.active_ontology_version,
            "active_runtime_id": self.active_runtime_id,
        }


@dataclass(frozen=True)
class UnresolvedConstraintState:
    unanswered_constraints: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    conflicting_evidence: tuple[str, ...] = ()
    unsatisfied_tool_postconditions: tuple[str, ...] = ()
    answer_instability: float = 0.0

    def validate(self) -> None:
        for field, values in (
            ("unanswered_constraints", self.unanswered_constraints),
            ("missing_evidence", self.missing_evidence),
            ("conflicting_evidence", self.conflicting_evidence),
            ("unsatisfied_tool_postconditions", self.unsatisfied_tool_postconditions),
        ):
            _nonempty_strings(values, field)
        if not 0.0 <= self.answer_instability <= 1.0:
            raise RecurrentContractError("answer instability must be in [0, 1]")

    def clear(self, maximum_instability: float) -> bool:
        self.validate()
        return not any(
            (
                self.unanswered_constraints,
                self.missing_evidence,
                self.conflicting_evidence,
                self.unsatisfied_tool_postconditions,
            )
        ) and self.answer_instability <= maximum_instability

    def to_dict(self) -> dict[str, Any]:
        return {
            "unanswered_constraints": list(self.unanswered_constraints),
            "missing_evidence": list(self.missing_evidence),
            "conflicting_evidence": list(self.conflicting_evidence),
            "unsatisfied_tool_postconditions": list(
                self.unsatisfied_tool_postconditions
            ),
            "answer_instability": self.answer_instability,
        }


@dataclass(frozen=True)
class HighLevelState:
    objective: str
    decomposition: tuple[str, ...]
    unresolved_constraints: tuple[str, ...]
    progress_estimate: float

    def validate(self) -> None:
        if not self.objective:
            raise RecurrentContractError("high-level objective is required")
        _nonempty_strings(self.decomposition, "decomposition")
        _nonempty_strings(self.unresolved_constraints, "unresolved_constraints")
        if not 0.0 <= self.progress_estimate <= 1.0:
            raise RecurrentContractError("progress estimate must be in [0, 1]")

    @property
    def state_hash(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "decomposition": list(self.decomposition),
            "unresolved_constraints": list(self.unresolved_constraints),
            "progress_estimate": self.progress_estimate,
        }


@dataclass(frozen=True)
class LowLevelState:
    local_computation: str
    candidate_answer: Any
    retrieval_requests: tuple[str, ...]
    tool_requests: tuple[str, ...]
    evidence_references: tuple[str, ...]

    def validate(self) -> None:
        if not self.local_computation:
            raise RecurrentContractError("low-level local computation is required")
        _nonempty_strings(self.retrieval_requests, "retrieval_requests")
        _nonempty_strings(self.tool_requests, "tool_requests")
        _nonempty_strings(self.evidence_references, "evidence_references")

    @property
    def state_hash(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_computation": self.local_computation,
            "candidate_answer": self.candidate_answer,
            "retrieval_requests": list(self.retrieval_requests),
            "tool_requests": list(self.tool_requests),
            "evidence_references": list(self.evidence_references),
        }


@dataclass(frozen=True)
class EvidenceOperation:
    operation_id: str
    operation_type: str
    request: str
    journal_id: str
    supporting_event_ids: tuple[str, ...]

    def validate(self) -> None:
        if self.operation_type not in {"retrieval", "tool"}:
            raise RecurrentContractError("operation type must be retrieval or tool")
        if not self.operation_id or not self.request or not self.journal_id:
            raise RecurrentContractError(
                "every evidence/tool operation requires request and journal identity"
            )
        _nonempty_strings(self.supporting_event_ids, "supporting_event_ids")

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "request": self.request,
            "journal_id": self.journal_id,
            "supporting_event_ids": list(self.supporting_event_ids),
        }


def decide_halt(
    learned_decision: str,
    residual: UnresolvedConstraintState,
    *,
    budget_exhausted: bool,
    maximum_answer_instability: float,
) -> str:
    if learned_decision not in {"continue", "stop"}:
        raise RecurrentContractError("learned halt decision must be continue or stop")
    clear = residual.clear(maximum_answer_instability)
    if budget_exhausted:
        return "resolved_at_budget" if clear else "unresolved_budget"
    if learned_decision == "stop" and clear:
        return "resolved"
    return "continue"


@dataclass(frozen=True)
class TraceStep:
    step_id: str
    macro_step: int
    micro_step: int
    high_state_in_hash: str
    high_state_out_hash: str
    low_state_hash: str
    injected_problem_hash: str
    injected_evidence_bundle_hash: str
    revalidated_view_ids: tuple[str, ...]
    learned_decision: str
    residual: UnresolvedConstraintState
    halt_decision: str
    operations: tuple[EvidenceOperation, ...]
    flop_delta: int
    remaining_flops: int
    remaining_macro_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "macro_step": self.macro_step,
            "micro_step": self.micro_step,
            "high_state_in_hash": self.high_state_in_hash,
            "high_state_out_hash": self.high_state_out_hash,
            "low_state_hash": self.low_state_hash,
            "injected_problem_hash": self.injected_problem_hash,
            "injected_evidence_bundle_hash": self.injected_evidence_bundle_hash,
            "revalidated_view_ids": list(self.revalidated_view_ids),
            "learned_decision": self.learned_decision,
            "residual": self.residual.to_dict(),
            "halt_decision": self.halt_decision,
            "operations": [operation.to_dict() for operation in self.operations],
            "flop_delta": self.flop_delta,
            "remaining_flops": self.remaining_flops,
            "remaining_macro_steps": self.remaining_macro_steps,
        }


@dataclass(frozen=True)
class ReasoningTrace:
    trace_id: str
    arm_id: str
    task_id: str
    problem_hash: str
    evidence_bundle_hash: str
    evidence_view_ids: tuple[str, ...]
    initial_high_state_hash: str
    budget: ComputeBudget
    steps: tuple[TraceStep, ...]
    maximum_answer_instability: float
    schema_version: int = TRACE_SCHEMA_VERSION
    interface_version: str = RECURRENT_INTERFACE_VERSION

    def validate(self) -> None:
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise RecurrentContractError("unsupported recurrent trace schema")
        if self.interface_version != RECURRENT_INTERFACE_VERSION:
            raise RecurrentContractError("unsupported recurrent interface version")
        if not all((self.trace_id, self.arm_id, self.task_id, self.problem_hash)):
            raise RecurrentContractError("trace identity is incomplete")
        self.budget.validate()
        expected_views = set(_nonempty_strings(self.evidence_view_ids, "evidence_view_ids"))
        if not self.steps:
            raise RecurrentContractError("a completed trace requires at least one step")
        previous_high_hash = self.initial_high_state_hash
        previous_position = (0, 0)
        total_flops = 0
        retrieval_calls = 0
        tool_calls = 0
        seen_step_ids: set[str] = set()
        for index, step in enumerate(self.steps):
            if not step.step_id or step.step_id in seen_step_ids:
                raise RecurrentContractError("trace step IDs must be unique and non-empty")
            seen_step_ids.add(step.step_id)
            position = (step.macro_step, step.micro_step)
            if (
                step.macro_step < 1
                or step.micro_step < 1
                or position <= previous_position
                or step.macro_step > self.budget.maximum_macro_steps
                or step.micro_step > self.budget.maximum_micro_steps_per_macro
            ):
                raise RecurrentContractError("trace macro/micro positions are invalid")
            previous_position = position
            if step.high_state_in_hash != previous_high_hash:
                raise RecurrentContractError("high-level recurrent state chain is broken")
            previous_high_hash = step.high_state_out_hash
            if step.injected_problem_hash != self.problem_hash:
                raise RecurrentContractError("a recurrence omitted the immutable problem")
            if step.injected_evidence_bundle_hash != self.evidence_bundle_hash:
                raise RecurrentContractError("a recurrence substituted its evidence bundle")
            if set(step.revalidated_view_ids) != expected_views:
                raise RecurrentContractError(
                    "a recurrence did not revalidate the complete evidence bundle"
                )
            if step.flop_delta < 0 or step.remaining_flops < 0 or step.remaining_macro_steps < 0:
                raise RecurrentContractError("trace compute counters cannot be negative")
            total_flops += step.flop_delta
            for operation in step.operations:
                operation.validate()
                if operation.operation_type == "retrieval":
                    retrieval_calls += 1
                else:
                    tool_calls += 1
            budget_exhausted = (
                step.remaining_flops == 0 or step.remaining_macro_steps == 0
            )
            expected_halt = decide_halt(
                step.learned_decision,
                step.residual,
                budget_exhausted=budget_exhausted,
                maximum_answer_instability=self.maximum_answer_instability,
            )
            if step.halt_decision != expected_halt:
                raise RecurrentContractError(
                    "trace halt decision bypasses the learned/external double gate"
                )
            if index < len(self.steps) - 1 and step.halt_decision != "continue":
                raise RecurrentContractError("trace continues after a halt decision")
        if self.steps[-1].halt_decision == "continue":
            raise RecurrentContractError("completed trace has no terminal halt")
        if len(self.steps) > self.budget.maximum_total_micro_steps:
            raise RecurrentContractError("trace exceeds total micro-step budget")
        if total_flops > self.budget.maximum_inference_flops:
            raise RecurrentContractError("trace exceeds its inference FLOP budget")
        if retrieval_calls > self.budget.maximum_retrieval_calls:
            raise RecurrentContractError("trace exceeds its retrieval-call budget")
        if tool_calls > self.budget.maximum_tool_calls:
            raise RecurrentContractError("trace exceeds its tool-call budget")

    @property
    def trace_hash(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "interface_version": self.interface_version,
            "trace_id": self.trace_id,
            "arm_id": self.arm_id,
            "task_id": self.task_id,
            "problem_hash": self.problem_hash,
            "evidence_bundle_hash": self.evidence_bundle_hash,
            "evidence_view_ids": list(self.evidence_view_ids),
            "initial_high_state_hash": self.initial_high_state_hash,
            "budget": self.budget.to_dict(),
            "maximum_answer_instability": self.maximum_answer_instability,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class ReasonerOutput:
    output_id: str
    answer: Any
    action: Any
    support: SupportManifest
    unresolved_residuals: UnresolvedConstraintState
    halt_reason: str
    executed_macro_steps: int
    executed_micro_steps: int
    problem_hash: str
    evidence_bundle_hash: str
    trace_hash: str

    def validate(self, trace: ReasoningTrace) -> None:
        trace.validate()
        if self.halt_reason not in HALT_REASONS - {"continue"}:
            raise RecurrentContractError("reasoner output requires a terminal halt reason")
        if self.halt_reason != trace.steps[-1].halt_decision:
            raise RecurrentContractError("output halt reason does not match its trace")
        if self.problem_hash != trace.problem_hash:
            raise RecurrentContractError("output problem identity does not match its trace")
        if self.evidence_bundle_hash != trace.evidence_bundle_hash:
            raise RecurrentContractError("output evidence identity does not match its trace")
        if self.trace_hash != trace.trace_hash:
            raise RecurrentContractError("output trace hash is incorrect")
        if self.executed_micro_steps != len(trace.steps):
            raise RecurrentContractError("output micro-step count is incorrect")
        macros = {step.macro_step for step in trace.steps}
        if self.executed_macro_steps != len(macros):
            raise RecurrentContractError("output macro-step count is incorrect")
        self.support.validate_shape()
        self.unresolved_residuals.validate()
        if self.halt_reason.startswith("resolved") and not self.unresolved_residuals.clear(
            trace.maximum_answer_instability
        ):
            raise RecurrentContractError("resolved output retains external residuals")


def audit_comparison(
    arm_values: list[dict[str, Any]],
    *,
    maximum_parameter_spread: float,
    maximum_flop_budget_spread: float,
) -> dict[str, Any]:
    required_ids = (
        "encoder_id",
        "decoder_id",
        "training_data_id",
        "evidence_interface_id",
        "optimizer_budget_id",
        "maximum_retrieval_calls",
        "maximum_tool_calls",
    )
    mismatches: dict[str, Any] = {}
    if len(arm_values) != 3 or len({arm.get("arm_id") for arm in arm_values}) != 3:
        mismatches["arms"] = "exactly three unique comparison arms are required"
    for field in required_ids:
        values = {arm.get(field) for arm in arm_values}
        if len(values) != 1:
            mismatches[field] = sorted(map(str, values))
    parameters = [int(arm["registered_parameter_count"]) for arm in arm_values]
    parameter_spread = (max(parameters) - min(parameters)) / max(parameters)
    if parameter_spread > maximum_parameter_spread:
        mismatches["parameter_spread"] = parameter_spread
    flops = [int(arm["maximum_inference_flops_per_example"]) for arm in arm_values]
    flop_spread = (max(flops) - min(flops)) / max(flops)
    if flop_spread > maximum_flop_budget_spread:
        mismatches["flop_budget_spread"] = flop_spread
    return {
        "arm_ids": [arm["arm_id"] for arm in arm_values],
        "parameter_spread": parameter_spread,
        "flop_budget_spread": flop_spread,
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def audit_depth_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    train_macro = tuple(protocol["training_macro_depth_distribution"]["support"])
    train_micro = tuple(protocol["training_micro_depth_distribution"]["support"])
    evaluation = tuple(protocol["evaluation_macro_depths"])
    unseen = tuple(protocol["unseen_evaluation_macro_depths"])
    checks = {
        "training_macro_depth_randomized": len(set(train_macro)) > 1,
        "training_micro_depth_randomized": len(set(train_micro)) > 1,
        "unseen_depths_disjoint": set(unseen).isdisjoint(train_macro),
        "unseen_depths_evaluated": set(unseen) <= set(evaluation),
        "maximum_macro_consistent": max(evaluation)
        == protocol["maximum_macro_steps"],
        "maximum_micro_consistent": (
            protocol["maximum_total_micro_steps"]
            == protocol["maximum_macro_steps"]
            * protocol["maximum_micro_steps_per_macro"]
        ),
        "initial_state_randomized": (
            protocol["initial_state_distribution"]["zero_probability"] == 0.5
            and protocol["initial_state_distribution"]["gaussian_probability"]
            == 0.5
            and protocol["initial_state_distribution"]["seeded_per_example"]
        ),
        "input_recall_explicit": "every macro and micro recurrence"
        in protocol["input_recall"],
        "gradient_policy_explicit": bool(protocol["gradient_truncation"]),
    }
    return {"checks": checks, "passed": all(checks.values())}


def protected_region_coverage(
    required_regions: tuple[str, ...],
    bundle: EvidenceBundle,
) -> dict[str, float]:
    covered = {
        region for view in bundle.views for region in view.protected_regions
    }
    return {
        region: 1.0 if region in covered else 0.0 for region in required_regions
    }
