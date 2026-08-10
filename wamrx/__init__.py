"""Grounded multiview memory kernel for WAM-RX milestones 1 and 2.

The package deliberately contains no model code.  It establishes the authority,
lineage, deletion, and audit contracts that later memory and reasoning layers
must use.
"""

from .analytic import (
    AnalyticCompileError,
    AnalyticMemory,
    AnalyticQueryResult,
    AnalyticRow,
    CandidateField,
)
from .artifacts import (
    ArtifactCompatibilityPolicy,
    ArtifactEnvelope,
    ArtifactStamp,
    InvalidArtifactError,
    SupportManifest,
)
from .contracts import ContractError, MechanismDeclaration, load_contracts
from .events import Event, SpeechAct
from .evaluation import CompileAdequacyReport, QueryProbe, measure_compile_adequacy
from .belief_graph import (
    BeliefGraph,
    BeliefGraphCompileError,
    ConstraintEvaluation,
    ConstraintRequirement,
    ConstraintSelection,
)
from .grounding import GroundingAuditor, GroundingError, GroundingReport
from .resolver import ResolvedSnapshot, resolve
from .retrieval import HybridRetrievalIndex, RetrievalResult
from .recurrent import (
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
from .recurrent_tasks import (
    RecurrentTask,
    RecurrentTaskError,
    generate_family,
    load_split_registry,
    verify_frozen_splits,
)
from .store import AppendOnlyEventStore, EventConflictError

__all__ = [
    "AppendOnlyEventStore",
    "AnalyticCompileError",
    "AnalyticMemory",
    "AnalyticQueryResult",
    "AnalyticRow",
    "ArtifactCompatibilityPolicy",
    "ArtifactEnvelope",
    "ArtifactStamp",
    "BeliefGraph",
    "BeliefGraphCompileError",
    "CandidateField",
    "ContractError",
    "CompileAdequacyReport",
    "ComputeBudget",
    "ComputeRecord",
    "ConstraintEvaluation",
    "ConstraintRequirement",
    "ConstraintSelection",
    "Event",
    "EventConflictError",
    "EvidenceBundle",
    "EvidenceOperation",
    "EvidenceViewReference",
    "GroundingAuditor",
    "GroundingError",
    "GroundingReport",
    "HighLevelState",
    "InvalidArtifactError",
    "HybridRetrievalIndex",
    "MechanismDeclaration",
    "LowLevelState",
    "QueryProbe",
    "ReasonerOutput",
    "ReasoningTrace",
    "RecurrentContractError",
    "RecurrentTask",
    "RecurrentTaskError",
    "RetrievalResult",
    "ResolvedSnapshot",
    "SpeechAct",
    "SupportManifest",
    "TraceStep",
    "UnresolvedConstraintState",
    "audit_comparison",
    "audit_depth_protocol",
    "decide_halt",
    "generate_family",
    "load_contracts",
    "measure_compile_adequacy",
    "load_split_registry",
    "protected_region_coverage",
    "resolve",
    "verify_frozen_splits",
]
