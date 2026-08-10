"""Evidence-bound adaptive execution for the small MLX recurrent reasoners."""

from __future__ import annotations

import json
from typing import Any

import mlx.core as mx

from .artifacts import SupportManifest
from .canonical import sha256_json
from .recurrent import (
    ComputeBudget,
    ComputeRecord,
    EvidenceBundle,
    EvidenceViewReference,
    HighLevelState,
    LowLevelState,
    ReasonerOutput,
    ReasoningTrace,
    TraceStep,
    UnresolvedConstraintState,
    decide_halt,
)
from .recurrent_model import ByteCodec, RecurrentReasoner, parameter_count
from .recurrent_tasks import RecurrentTask

EXECUTOR_VERSION = "wamrx-recurrent-executor-v1"


class RecurrentExecutionError(ValueError):
    pass


def task_evidence_bundle(task: RecurrentTask) -> EvidenceBundle:
    views = []
    for value in task.problem["evidence_bundle"]["views"]:
        views.append(
            EvidenceViewReference(
                view_type=str(value["view_type"]),
                artifact_id=str(value["artifact_id"]),
                content_hash=str(value["content_hash"]),
                ledger_frontier_sequence=int(value["ledger_frontier_sequence"]),
                ledger_frontier_hash=str(value["ledger_frontier_hash"]),
                ontology_version=str(value["ontology_version"]),
                component_versions=dict(value["component_versions"]),
                support_event_ids=tuple(value["support_event_ids"]),
                protected_regions=tuple(value["regions"]),
            )
        )
    bundle = EvidenceBundle(
        bundle_id=f"bundle:{task.task_id}",
        problem_hash=task.problem_hash,
        views=tuple(views),
        active_ontology_version="v1",
        active_runtime_id=EXECUTOR_VERSION,
    )
    bundle.validate(required_regions=task.protected_regions)
    return bundle


def _array_hash(value: mx.array) -> str:
    mx.eval(value)
    return sha256_json(value.tolist())


def _prediction(logits: mx.array, codec: ByteCodec) -> Any:
    mx.eval(logits)
    tokens = mx.argmax(logits, axis=-1)[0].tolist()
    return codec.decode_answer(tokens)


def _residual(
    task: RecurrentTask,
    prediction: Any,
    previous_prediction: Any,
) -> UnresolvedConstraintState:
    correct = prediction == task.expected
    stable = prediction is not None and prediction == previous_prediction
    return UnresolvedConstraintState(
        unanswered_constraints=() if correct else ("executable_answer_verifier",),
        missing_evidence=(),
        conflicting_evidence=(),
        unsatisfied_tool_postconditions=(),
        answer_instability=0.0 if stable else 1.0,
    )


def _semantic_high_state(
    task: RecurrentTask,
    residual: UnresolvedConstraintState,
    progress: float,
) -> HighLevelState:
    unresolved = (
        *residual.unanswered_constraints,
        *residual.missing_evidence,
        *residual.conflicting_evidence,
        *residual.unsatisfied_tool_postconditions,
    )
    return HighLevelState(
        objective=str(task.problem.get("instruction", task.task_id)),
        decomposition=(f"solve:{task.family}", task.generalization_axis),
        unresolved_constraints=tuple(unresolved),
        progress_estimate=min(1.0, max(0.0, progress)),
    )


def _consumed_flops(
    model: RecurrentReasoner,
    *,
    macro_step: int,
    micro_step: int,
    configured_micro_steps: int,
    readout_count: int,
) -> int:
    config = model.config
    d = config.hidden_dimensions
    ff = config.feedforward_dimensions
    constant = (
        config.maximum_input_tokens * d * 2
        + config.vocabulary_size * d * 2
        + 2 * d * d
    )
    block = 4 * d * ff
    injection = model.capacity_match.estimated_flops() + d
    coda = 2 * config.maximum_output_tokens * d * config.vocabulary_size
    halt = 2 * d
    if readout_count < 0:
        raise RecurrentExecutionError("readout count cannot be negative")
    if model.arm_id == "fixed-depth-v1":
        block_calls = macro_step
        injection_calls = macro_step
    elif model.arm_id == "flat-recurrent-v1":
        block_calls = config.flat_core_blocks * macro_step
        injection_calls = macro_step
    else:
        completed_macros = macro_step - 1
        block_calls = (
            completed_macros
            * (
                config.hierarchical_high_blocks
                + config.hierarchical_low_blocks * configured_micro_steps
            )
            + config.hierarchical_high_blocks
            + config.hierarchical_low_blocks * micro_step
        )
        injection_calls = (
            completed_macros * (1 + configured_micro_steps) + 1 + micro_step
        )
    return int(
        constant
        + block * block_calls
        + injection * injection_calls
        + (coda + halt) * readout_count
    )


def execute_task(
    model: RecurrentReasoner,
    codec: ByteCodec,
    task: RecurrentTask,
    *,
    budget: ComputeBudget,
    maximum_answer_instability: float,
    maximum_macro_steps: int,
    micro_steps: int = 1,
    execution_policy: str = "adaptive",
) -> tuple[ReasonerOutput, ReasoningTrace, ComputeRecord]:
    if maximum_macro_steps < 1 or micro_steps < 1:
        raise RecurrentExecutionError("execution step counts must be positive")
    if execution_policy not in {"adaptive", "maximum_depth", "final_depth"}:
        raise RecurrentExecutionError(
            "execution policy must be adaptive, maximum_depth, or final_depth"
        )
    bundle = task_evidence_bundle(task)
    problem_tokens, _ = codec.encode_problem(task.problem)
    tokens = mx.array([problem_tokens], dtype=mx.int32)
    context = model.encode(tokens)
    zero = mx.zeros_like(context)
    view_ids = tuple(view.artifact_id for view in bundle.views)
    previous_prediction = None
    initial_residual = UnresolvedConstraintState(
        unanswered_constraints=("executable_answer_verifier",),
        answer_instability=1.0,
    )
    semantic_high = _semantic_high_state(task, initial_residual, 0.0)
    initial_high_hash = semantic_high.state_hash
    steps = []
    prediction = None
    final_residual = initial_residual

    if model.arm_id == "fixed-depth-v1":
        maximum_macro_steps = 4
        outputs = model(tokens, macro_steps=4, micro_steps=1)
        states_by_macro = [(index + 1, 1, state) for index, state in enumerate(outputs["states"])]
    elif model.arm_id == "flat-recurrent-v1":
        state = zero
        states_by_macro = []
        for macro in range(1, maximum_macro_steps + 1):
            state = model.flat_step(context, state)
            states_by_macro.append((macro, 1, state))
    else:
        high = zero
        low = zero
        states_by_macro = []
        for macro in range(1, maximum_macro_steps + 1):
            high, low, micro_states = model.hierarchical_step(
                context, high, low, micro_steps
            )
            states_by_macro.extend(
                (macro, micro + 1, state)
                for micro, state in enumerate(micro_states)
            )

    total_positions = len(states_by_macro)
    for index, (macro, micro, neural_state) in enumerate(states_by_macro):
        bundle.validate(required_regions=task.protected_regions)
        last_registered_position = index == total_positions - 1
        performs_readout = execution_policy != "final_depth" or last_registered_position
        if performs_readout:
            logits, halt_logit = model.state_outputs(neural_state)
            prediction = _prediction(logits, codec)
            final_residual = _residual(task, prediction, previous_prediction)
            previous_prediction = prediction
        else:
            halt_logit = None
            prediction = None
            final_residual = UnresolvedConstraintState(
                unanswered_constraints=("final_depth_readout_pending",),
                answer_instability=1.0,
            )
        is_fixed = model.arm_id == "fixed-depth-v1"
        if execution_policy in {"maximum_depth", "final_depth"}:
            learned_decision = (
                "continue" if index < total_positions - 1 else "stop"
            )
        else:
            learned_decision = (
                "continue"
                if is_fixed and index < total_positions - 1
                else "stop"
                if is_fixed
                else "stop"
                if float(halt_logit[0, 0]) >= 0.0
                else "continue"
            )
        consumed_flops = _consumed_flops(
            model,
            macro_step=macro,
            micro_step=micro,
            configured_micro_steps=(1 if is_fixed else micro_steps),
            readout_count=(
                index + 1
                if execution_policy != "final_depth"
                else 1
                if last_registered_position
                else 0
            ),
        )
        if consumed_flops > budget.maximum_inference_flops:
            raise RecurrentExecutionError(
                "requested recurrent position exceeds the common FLOP budget"
            )
        remaining_flops = budget.maximum_inference_flops - consumed_flops
        remaining_macro = max(0, maximum_macro_steps - macro)
        budget_exhausted = (
            remaining_flops == 0 or remaining_macro == 0 or last_registered_position
        )
        halt = decide_halt(
            learned_decision,
            final_residual,
            budget_exhausted=budget_exhausted,
            maximum_answer_instability=maximum_answer_instability,
        )
        low = LowLevelState(
            local_computation=f"neural_state_hash:{_array_hash(neural_state)}",
            candidate_answer=prediction,
            retrieval_requests=(),
            tool_requests=(),
            evidence_references=view_ids,
        )
        next_high = _semantic_high_state(
            task,
            final_residual,
            (index + 1) / total_positions,
        )
        previous_consumed = (
            0
            if not steps
            else budget.maximum_inference_flops - steps[-1].remaining_flops
        )
        steps.append(
            TraceStep(
                step_id=f"{task.task_id}:m{macro}:u{micro}",
                macro_step=macro,
                micro_step=micro,
                high_state_in_hash=semantic_high.state_hash,
                high_state_out_hash=next_high.state_hash,
                low_state_hash=low.state_hash,
                injected_problem_hash=task.problem_hash,
                injected_evidence_bundle_hash=bundle.bundle_hash,
                revalidated_view_ids=view_ids,
                learned_decision=learned_decision,
                residual=final_residual,
                halt_decision=halt,
                operations=(),
                flop_delta=consumed_flops - previous_consumed,
                remaining_flops=remaining_flops,
                remaining_macro_steps=remaining_macro,
            )
        )
        semantic_high = next_high
        if halt != "continue":
            break

    trace = ReasoningTrace(
        trace_id=f"trace:{model.arm_id}:{task.task_id}",
        arm_id=model.arm_id,
        task_id=task.task_id,
        problem_hash=task.problem_hash,
        evidence_bundle_hash=bundle.bundle_hash,
        evidence_view_ids=view_ids,
        initial_high_state_hash=initial_high_hash,
        budget=budget,
        steps=tuple(steps),
        maximum_answer_instability=maximum_answer_instability,
    )
    trace.validate()
    evidence_ids = sorted(
        {
            event_id
            for view in bundle.views
            for event_id in view.support_event_ids
        }
    )
    output = ReasonerOutput(
        output_id=f"output:{model.arm_id}:{task.task_id}",
        answer=prediction,
        action=None,
        support=SupportManifest.create(
            supporting_event_ids=evidence_ids,
            candidate_event_ids=evidence_ids,
        ),
        unresolved_residuals=final_residual,
        halt_reason=steps[-1].halt_decision,
        executed_macro_steps=len({step.macro_step for step in steps}),
        executed_micro_steps=len(steps),
        problem_hash=task.problem_hash,
        evidence_bundle_hash=bundle.bundle_hash,
        trace_hash=trace.trace_hash,
    )
    output.validate(trace)
    final_flops = budget.maximum_inference_flops - steps[-1].remaining_flops
    compute = ComputeRecord(
        arm_id=model.arm_id,
        task_id=task.task_id,
        parameter_count=parameter_count(model),
        inference_flops=final_flops,
        macro_steps=output.executed_macro_steps,
        micro_steps=output.executed_micro_steps,
        retrieval_calls=0,
        tool_calls=0,
        training_examples_seen=0,
        optimizer_updates=0,
    )
    compute.validate(budget)
    return output, trace, compute
