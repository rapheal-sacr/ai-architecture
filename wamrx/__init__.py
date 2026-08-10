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
    "ConstraintEvaluation",
    "ConstraintRequirement",
    "ConstraintSelection",
    "Event",
    "EventConflictError",
    "GroundingAuditor",
    "GroundingError",
    "GroundingReport",
    "InvalidArtifactError",
    "HybridRetrievalIndex",
    "MechanismDeclaration",
    "QueryProbe",
    "RetrievalResult",
    "ResolvedSnapshot",
    "SpeechAct",
    "SupportManifest",
    "load_contracts",
    "measure_compile_adequacy",
    "resolve",
]
