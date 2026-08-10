"""Milestone-1 kernel for WAM-RX.

The package deliberately contains no model code.  It establishes the authority,
lineage, deletion, and audit contracts that later memory and reasoning layers
must use.
"""

from .artifacts import (
    ArtifactEnvelope,
    ArtifactStamp,
    InvalidArtifactError,
    SupportManifest,
)
from .contracts import ContractError, MechanismDeclaration, load_contracts
from .events import Event, SpeechAct
from .evaluation import CompileAdequacyReport, QueryProbe, measure_compile_adequacy
from .resolver import ResolvedSnapshot, resolve
from .retrieval import HybridRetrievalIndex, RetrievalResult
from .store import AppendOnlyEventStore, EventConflictError

__all__ = [
    "AppendOnlyEventStore",
    "ArtifactEnvelope",
    "ArtifactStamp",
    "ContractError",
    "CompileAdequacyReport",
    "Event",
    "EventConflictError",
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
