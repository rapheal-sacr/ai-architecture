"""E0.12 -- pre-model recurrent-reasoner contract and assay validation.

CLAIM UNDER TEST
    The Milestone 3 comparison, recurrent interface, depth protocol, external
    halting gate, task splits, compute accounting, and promotion thresholds are
    frozen and capable of invalidating their registered manipulations before
    any model is implemented or trained.

SCOPE
    This experiment does not implement a neural model and produces no evidence
    that recurrence, hierarchy, or adaptive halting improves task accuracy.

REGISTERED KILL CRITERIA
    K1  the recurrent contract has a missing or ambiguous mechanism declaration.
    K2  the nominal three-arm comparison is not parameter/FLOP/data/interface matched.
    K3  a registered comparison confound is not detected before training.
    K4  the depth protocol lacks random train depths or genuinely unseen eval depths.
    K5  any deterministic frozen split content hash changes.
    K6  any OOD length, graph structure, or composition overlaps training.
    K7  the recurrent input/evidence frontier cannot be bound to every trace step.
    K8  a completed trace bypasses the learned/external double halt gate.
    K9  compute, retrieval, or tool budgets can be exceeded without failure.
    K10 an evidence/tool operation can appear without a journal identifier.
    K11 trace or compute schemas omit an executable contract field.
    K12 promotion criteria omit a numeric tolerance or confidence requirement.

MANIPULATION CHECKS
    M1  disabling problem/evidence reinjection MUST invalidate a trace.
    M2  one fixed training loop count MUST fail the depth-protocol audit.
    M3  corrupting the high-level state chain MUST invalidate a trace.
    M4  removing external halt enforcement MUST invalidate a residual-bearing halt.
    M5  contradictory evidence MUST keep the external residual active.
    M6  OOD tasks MUST exceed the common training depth or training axis.
    M7  stale/incompatible evidence MUST fail before reasoner execution.
    M8  removing operations evidence MUST fail the protected-region gate.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wamrx.artifacts import SupportManifest  # noqa: E402
from wamrx.canonical import sha256_json  # noqa: E402
from wamrx.contracts import load_contracts  # noqa: E402
from wamrx.recurrent import (  # noqa: E402
    ComputeBudget,
    ComputeRecord,
    EvidenceBundle,
    EvidenceOperation,
    EvidenceViewReference,
    HighLevelState,
    LowLevelState,
    ReasonerOutput,
    ReasoningTrace,
    RecurrentContractError,
    TraceStep,
    UnresolvedConstraintState,
    audit_comparison,
    audit_depth_protocol,
    decide_halt,
    protected_region_coverage,
)
from wamrx.recurrent_tasks import (  # noqa: E402
    FAMILIES,
    generate_family,
    load_split_registry,
    verify_frozen_splits,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contracts" / "wamrx_recurrent_reasoner.json"
SPLIT_PATH = ROOT / "contracts" / "wamrx_recurrent_splits.json"


def raises_contract_error(function) -> bool:
    try:
        function()
    except RecurrentContractError:
        return True
    return False


def evidence_view(
    view_type: str,
    artifact_id: str,
    region: str,
    *,
    ontology_version: str = "v1",
    compatible: bool = True,
    support_usable: bool = True,
) -> EvidenceViewReference:
    return EvidenceViewReference(
        view_type=view_type,
        artifact_id=artifact_id,
        content_hash=sha256_json({"artifact_id": artifact_id}),
        ledger_frontier_sequence=20,
        ledger_frontier_hash=sha256_json({"frontier": 20}),
        ontology_version=ontology_version,
        component_versions={f"{view_type}_compiler": "v1"},
        support_event_ids=(f"event:{artifact_id}",),
        protected_regions=(region,),
        compatible_with_runtime=compatible,
        support_usable=support_usable,
    )


def valid_fixture() -> tuple[
    EvidenceBundle,
    ComputeBudget,
    ReasoningTrace,
    ReasonerOutput,
    ComputeRecord,
]:
    problem_hash = sha256_json({"problem": "select the eligible candidate"})
    views = (
        evidence_view("analytic", "analytic-v1", "finance"),
        evidence_view("graph", "graph-v1", "operations"),
    )
    bundle = EvidenceBundle(
        bundle_id="bundle-e0.12",
        problem_hash=problem_hash,
        views=views,
        active_ontology_version="v1",
        active_runtime_id="wamrx-m3-runtime-v1",
    )
    bundle.validate(required_regions=("finance", "operations"))
    budget = ComputeBudget(
        maximum_inference_flops=200_000_000,
        maximum_macro_steps=12,
        maximum_micro_steps_per_macro=4,
        maximum_total_micro_steps=48,
        maximum_retrieval_calls=4,
        maximum_tool_calls=4,
    )
    high_initial = HighLevelState(
        objective="select a candidate satisfying all constraints",
        decomposition=("retrieve constraints", "evaluate candidates"),
        unresolved_constraints=("latency",),
        progress_estimate=0.0,
    )
    high_after_retrieval = HighLevelState(
        objective=high_initial.objective,
        decomposition=high_initial.decomposition,
        unresolved_constraints=(),
        progress_estimate=0.5,
    )
    high_final = HighLevelState(
        objective=high_initial.objective,
        decomposition=high_initial.decomposition,
        unresolved_constraints=(),
        progress_estimate=1.0,
    )
    low_one = LowLevelState(
        local_computation="request missing latency evidence",
        candidate_answer=None,
        retrieval_requests=("candidate-c latency",),
        tool_requests=(),
        evidence_references=("analytic-v1", "graph-v1"),
    )
    low_two = LowLevelState(
        local_computation="all constraints satisfied",
        candidate_answer="candidate-c",
        retrieval_requests=(),
        tool_requests=("validate constraints",),
        evidence_references=("analytic-v1", "graph-v1"),
    )
    residual_one = UnresolvedConstraintState(
        missing_evidence=("latency",),
        answer_instability=0.4,
    )
    residual_clear = UnresolvedConstraintState(answer_instability=0.0)
    view_ids = tuple(view.artifact_id for view in views)
    step_one = TraceStep(
        step_id="step-1",
        macro_step=1,
        micro_step=1,
        high_state_in_hash=high_initial.state_hash,
        high_state_out_hash=high_after_retrieval.state_hash,
        low_state_hash=low_one.state_hash,
        injected_problem_hash=problem_hash,
        injected_evidence_bundle_hash=bundle.bundle_hash,
        revalidated_view_ids=view_ids,
        learned_decision="stop",
        residual=residual_one,
        halt_decision="continue",
        operations=(
            EvidenceOperation(
                operation_id="retrieve-1",
                operation_type="retrieval",
                request="candidate-c latency",
                journal_id="retrieval-journal-1",
                supporting_event_ids=("event:graph-v1",),
            ),
        ),
        flop_delta=40_000_000,
        remaining_flops=160_000_000,
        remaining_macro_steps=11,
    )
    step_two = TraceStep(
        step_id="step-2",
        macro_step=2,
        micro_step=1,
        high_state_in_hash=high_after_retrieval.state_hash,
        high_state_out_hash=high_final.state_hash,
        low_state_hash=low_two.state_hash,
        injected_problem_hash=problem_hash,
        injected_evidence_bundle_hash=bundle.bundle_hash,
        revalidated_view_ids=view_ids,
        learned_decision="stop",
        residual=residual_clear,
        halt_decision="resolved",
        operations=(
            EvidenceOperation(
                operation_id="tool-1",
                operation_type="tool",
                request="validate constraints",
                journal_id="tool-journal-1",
                supporting_event_ids=("event:analytic-v1", "event:graph-v1"),
            ),
        ),
        flop_delta=40_000_000,
        remaining_flops=120_000_000,
        remaining_macro_steps=10,
    )
    trace = ReasoningTrace(
        trace_id="trace-e0.12",
        arm_id="hierarchical-recurrent-v1",
        task_id="multiview/control/0000",
        problem_hash=problem_hash,
        evidence_bundle_hash=bundle.bundle_hash,
        evidence_view_ids=view_ids,
        initial_high_state_hash=high_initial.state_hash,
        budget=budget,
        steps=(step_one, step_two),
        maximum_answer_instability=0.02,
    )
    trace.validate()
    output = ReasonerOutput(
        output_id="output-e0.12",
        answer="candidate-c",
        action=None,
        support=SupportManifest.create(
            supporting_event_ids=["event:analytic-v1", "event:graph-v1"],
            candidate_event_ids=["event:analytic-v1", "event:graph-v1"],
        ),
        unresolved_residuals=residual_clear,
        halt_reason="resolved",
        executed_macro_steps=2,
        executed_micro_steps=2,
        problem_hash=problem_hash,
        evidence_bundle_hash=bundle.bundle_hash,
        trace_hash=trace.trace_hash,
    )
    output.validate(trace)
    compute = ComputeRecord(
        arm_id=trace.arm_id,
        task_id=trace.task_id,
        parameter_count=8_404_992,
        inference_flops=80_000_000,
        macro_steps=2,
        micro_steps=2,
        retrieval_calls=1,
        tool_calls=1,
        training_examples_seen=576,
        optimizer_updates=1000,
    )
    compute.validate(budget)
    return bundle, budget, trace, output, compute


def schema_is_complete(path: pathlib.Path, required: set[str]) -> bool:
    schema = json.loads(path.read_text())
    return (
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
        and required <= set(schema.get("required", ()))
        and schema.get("additionalProperties") is False
    )


def run() -> dict:
    contracts = load_contracts(CONTRACT_PATH)
    contract = json.loads(CONTRACT_PATH.read_text())
    split_registry = load_split_registry(SPLIT_PATH)
    split_report = verify_frozen_splits(split_registry)
    thresholds = contract["promotion_thresholds"]

    comparison = audit_comparison(
        contract["comparison_arms"],
        maximum_parameter_spread=thresholds[
            "maximum_parameter_count_relative_spread"
        ],
        maximum_flop_budget_spread=thresholds[
            "maximum_inference_flop_budget_relative_spread"
        ],
    )
    confounded_arms = copy.deepcopy(contract["comparison_arms"])
    confounded_arms[1]["training_data_id"] = "different-training-data"
    confounded_arms[2]["maximum_inference_flops_per_example"] += 1
    confound_audit = audit_comparison(
        confounded_arms,
        maximum_parameter_spread=thresholds[
            "maximum_parameter_count_relative_spread"
        ],
        maximum_flop_budget_spread=thresholds[
            "maximum_inference_flop_budget_relative_spread"
        ],
    )
    depth_audit = audit_depth_protocol(contract["depth_protocol"])
    fixed_depth_protocol = copy.deepcopy(contract["depth_protocol"])
    fixed_depth_protocol["training_macro_depth_distribution"]["support"] = [4]
    fixed_depth_protocol["training_micro_depth_distribution"]["support"] = [4]
    fixed_depth_detected = not audit_depth_protocol(fixed_depth_protocol)["passed"]

    bundle, budget, trace, output, compute = valid_fixture()

    no_reinjection = dataclasses.replace(
        trace,
        steps=(
            dataclasses.replace(
                trace.steps[0],
                injected_problem_hash=sha256_json({"wrong": "problem"}),
            ),
            trace.steps[1],
        ),
    )
    corrupt_high_state = dataclasses.replace(
        trace,
        steps=(
            trace.steps[0],
            dataclasses.replace(
                trace.steps[1],
                high_state_in_hash=sha256_json({"corrupted": "high-state"}),
            ),
        ),
    )
    no_external_halt = dataclasses.replace(
        trace,
        steps=(
            dataclasses.replace(trace.steps[0], halt_decision="resolved"),
            trace.steps[1],
        ),
    )
    contradictory_residual = UnresolvedConstraintState(
        conflicting_evidence=("candidate-c.latency",),
        answer_instability=0.3,
    )
    contradiction_blocks_halt = (
        decide_halt(
            "stop",
            contradictory_residual,
            budget_exhausted=False,
            maximum_answer_instability=0.02,
        )
        == "continue"
    )
    learned_continue_is_active = (
        decide_halt(
            "continue",
            UnresolvedConstraintState(),
            budget_exhausted=False,
            maximum_answer_instability=0.02,
        )
        == "continue"
    )

    stale_view = dataclasses.replace(
        bundle.views[0],
        ontology_version="v0",
        compatible_with_runtime=False,
    )
    stale_bundle = dataclasses.replace(bundle, views=(stale_view, bundle.views[1]))
    finance_only_bundle = dataclasses.replace(bundle, views=(bundle.views[0],))
    missing_region_coverage = protected_region_coverage(
        ("finance", "operations"), finance_only_bundle
    )

    excessive_compute = dataclasses.replace(
        compute,
        inference_flops=budget.maximum_inference_flops + 1,
    )
    unjournaled_operation = dataclasses.replace(
        trace.steps[0].operations[0], journal_id=""
    )

    trace_schema_complete = schema_is_complete(
        ROOT / "schemas" / "wamrx_recurrent_trace.schema.json",
        {
            "problem_hash",
            "evidence_bundle_hash",
            "evidence_view_ids",
            "initial_high_state_hash",
            "budget",
            "steps",
        },
    )
    compute_schema_complete = schema_is_complete(
        ROOT / "schemas" / "wamrx_compute_accounting.schema.json",
        {
            "parameter_count",
            "inference_flops",
            "macro_steps",
            "micro_steps",
            "retrieval_calls",
            "tool_calls",
            "training_examples_seen",
            "optimizer_updates",
        },
    )

    required_promotion_thresholds = {
        "maximum_parameter_count_relative_spread",
        "maximum_inference_flop_budget_relative_spread",
        "minimum_paired_training_seeds",
        "confidence_level",
        "minimum_ood_improvement_confidence_lower_bound",
        "maximum_in_distribution_accuracy_regression",
        "maximum_protected_region_accuracy_regression",
        "maximum_accuracy_decline_with_additional_loops",
        "minimum_adaptive_halting_median_compute_saving",
        "maximum_adaptive_halting_accuracy_loss",
        "minimum_high_low_winning_families",
        "maximum_high_low_regression_on_nonwinning_family",
        "maximum_unjournaled_evidence_or_tool_operations",
        "maximum_incompatible_or_tombstoned_artifacts_used",
    }
    numeric_thresholds_complete = required_promotion_thresholds <= set(thresholds) and all(
        isinstance(thresholds[key], (int, float))
        and not isinstance(thresholds[key], bool)
        for key in required_promotion_thresholds
    )

    train_examples = sum(
        len(generate_family("train", family, split_registry["splits"]["train"]))
        for family in FAMILIES
    )
    id_examples = sum(
        len(generate_family("id", family, split_registry["splits"]["id"]))
        for family in FAMILIES
    )
    ood_examples = sum(
        len(generate_family("ood", family, split_registry["splits"]["ood"]))
        for family in FAMILIES
    )

    checks = {
        "K1_six_mechanism_contract_complete": len(contracts) == 6,
        "K2_nominal_three_arm_comparison_matched": comparison["passed"],
        "K3_comparison_confound_detected_before_training": (
            not confound_audit["passed"]
            and "training_data_id" in confound_audit["mismatches"]
            and "flop_budget_spread" in confound_audit["mismatches"]
        ),
        "K4_random_depth_and_unseen_depth_protocol_valid": depth_audit["passed"],
        "K5_frozen_split_hashes_exact": (
            split_report["passed"]
            and sha256_json(split_registry)
            == contract["frozen_split_registry"]["content_hash"]
        ),
        "K6_ood_axes_disjoint_from_training": all(
            split_report["ood_axis_disjoint"].values()
        ),
        "K7_frontier_bound_interface_and_output_valid": (
            output.trace_hash == trace.trace_hash
            and output.evidence_bundle_hash == bundle.bundle_hash
        ),
        "K8_learned_and_external_halting_paths_active": (
            contradiction_blocks_halt and learned_continue_is_active
        ),
        "K9_hard_compute_budget_enforced": raises_contract_error(
            lambda: excessive_compute.validate(budget)
        ),
        "K10_evidence_operations_require_journals": raises_contract_error(
            unjournaled_operation.validate
        ),
        "K11_trace_and_compute_schemas_complete": (
            trace_schema_complete and compute_schema_complete
        ),
        "K12_numeric_promotion_thresholds_frozen": numeric_thresholds_complete,
    }
    manipulations = {
        "M1_missing_reinjection_invalidates_trace": raises_contract_error(
            no_reinjection.validate
        ),
        "M2_fixed_training_loop_detected": fixed_depth_detected,
        "M3_corrupt_high_state_invalidates_trace": raises_contract_error(
            corrupt_high_state.validate
        ),
        "M4_external_halt_removal_detected": raises_contract_error(
            no_external_halt.validate
        ),
        "M5_contradictory_evidence_blocks_halt": contradiction_blocks_halt,
        "M6_unseen_depth_and_axes_exercised": (
            split_report["maximum_required_reasoning_steps"]["algorithmic"]["ood"]
            > split_report["maximum_required_reasoning_steps"]["algorithmic"]["train"]
            and all(split_report["ood_axis_disjoint"].values())
        ),
        "M7_stale_incompatible_evidence_rejected": raises_contract_error(
            lambda: stale_bundle.validate(required_regions=("finance", "operations"))
        ),
        "M8_missing_protected_region_detected": (
            missing_region_coverage["operations"] == 0.0
            and raises_contract_error(
                lambda: finance_only_bundle.validate(
                    required_regions=("finance", "operations")
                )
            )
        ),
    }
    return {
        "experiment": "E0.12",
        "claim": "the recurrent-reasoner comparison and assay are frozen and manipulation-complete before model implementation",
        "scope": "pre-model structural assay validation only",
        "model_comparison_status": "NOT_RUN",
        "contract_count": len(contracts),
        "task_counts": {
            "train": train_examples,
            "id": id_examples,
            "ood": ood_examples,
            "total": train_examples + id_examples + ood_examples,
        },
        "comparison_audit": comparison,
        "depth_protocol": depth_audit,
        "frozen_split_hashes": split_report["hashes"],
        "ood_axes": {
            family: split_report["axes"][family]["ood"] for family in FAMILIES
        },
        "checks": checks,
        "manipulation_checks": manipulations,
        "verdict": (
            "PASS"
            if all(checks.values()) and all(manipulations.values())
            else "FAIL"
        ),
    }


def main() -> int:
    result = run()
    path = ROOT / "results" / "e0_12_recurrent_reasoner_assay.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
