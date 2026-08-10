"""Deterministic hybrid retrieval with complete, append-only selection journals."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .artifacts import (
    ArtifactCompatibilityPolicy,
    ArtifactEnvelope,
    ArtifactStamp,
    SupportManifest,
)
from .canonical import sha256_json
from .resolver import ResolvedRecord, resolve
from .store import AppendOnlyEventStore

TOKEN_RE = re.compile(r"[a-z0-9]+")
INDEX_VERSION = "wamrx-hybrid-v1"
QUERY_VERSION = "wamrx-query-v1"


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(text.lower()))


def _features(text: str) -> Counter[str]:
    compact = " ".join(_tokens(text))
    padded = f"  {compact}  "
    return Counter(padded[index : index + 3] for index in range(max(0, len(padded) - 2)))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


@dataclass(frozen=True)
class IndexedDocument:
    event_id: str
    text: str
    region: str
    metadata: dict[str, str]
    support: SupportManifest

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "text": self.text,
            "region": self.region,
            "metadata": self.metadata,
            "support": self.support.to_dict(),
        }


@dataclass(frozen=True)
class RetrievalHit:
    event_id: str
    text: str
    region: str
    score: float
    support: SupportManifest


@dataclass(frozen=True)
class RetrievalResult:
    query_id: str
    hits: tuple[RetrievalHit, ...]
    journal_hash: str
    boundary_margin: float | None


class HybridRetrievalIndex:
    def __init__(
        self,
        *,
        store: AppendOnlyEventStore,
        envelope: ArtifactEnvelope,
        documents: tuple[IndexedDocument, ...],
        disabled_records: dict[str, str],
        compatibility_policy: ArtifactCompatibilityPolicy,
    ) -> None:
        self.store = store
        self.envelope = envelope
        self.documents = documents
        self.disabled_records = disabled_records
        self.compatibility_policy = compatibility_policy

    @classmethod
    def build(
        cls,
        store: AppendOnlyEventStore,
        *,
        artifact_id: str,
        valid_at: str,
        include_event_ids: set[str] | None = None,
        base_weight_version: str = "none",
        ontology_version: str = "v1",
        verifier_version: str = "executable-v1",
    ) -> "HybridRetrievalIndex":
        snapshot = resolve(store, valid_at=valid_at)
        documents = []
        disabled: dict[str, str] = {}
        for record in snapshot.records:
            if record.usable:
                if include_event_ids is not None and record.event_id not in include_event_ids:
                    continue
                text = str(record.payload.get("text", "")).strip()
                if not text:
                    continue
                metadata = {
                    str(key): str(value)
                    for key, value in record.payload.get("metadata", {}).items()
                }
                manifest = SupportManifest.create(
                    supporting_event_ids=[record.event_id],
                    candidate_event_ids=[record.event_id, *record.control_event_ids],
                )
                documents.append(
                    IndexedDocument(
                        event_id=record.event_id,
                        text=text,
                        region=record.region,
                        metadata=dict(sorted(metadata.items())),
                        support=manifest,
                    )
                )
            else:
                disabled[record.event_id] = record.status
        documents.sort(key=lambda document: document.event_id)
        content = {
            "documents": [document.to_dict() for document in documents],
            "disabled_records": dict(sorted(disabled.items())),
            "valid_at": valid_at,
            "index_version": INDEX_VERSION,
        }
        support_ids = [document.event_id for document in documents]
        contradiction_ids = sorted(
            {
                event_id
                for record in snapshot.records
                if record.status in {"refuted", "retracted", "tombstone"}
                for event_id in record.control_event_ids
            }
        )
        artifact_candidates = sorted(
            {
                event_id
                for document in documents
                for event_id in document.support.candidate_event_ids
            }
            | set(contradiction_ids)
        )
        manifest = SupportManifest.create(
            supporting_event_ids=support_ids,
            contradicting_event_ids=contradiction_ids,
            candidate_event_ids=artifact_candidates,
            require_all_support=False,
            minimum_live_support=0,
        )
        build_config = {
            "valid_at": valid_at,
            "include_event_ids": sorted(include_event_ids) if include_event_ids else "all",
            "weights": {"lexical": 0.45, "semantic": 0.45, "phrase": 0.10},
        }
        stamp = ArtifactStamp.create(
            artifact_id=artifact_id,
            artifact_type="hybrid-retrieval-index",
            content=content,
            store=store,
            base_weight_version=base_weight_version,
            component_versions={
                "index": INDEX_VERSION,
                "query": QUERY_VERSION,
                "resolver": snapshot.resolver_version,
            },
            ontology_version=ontology_version,
            verifier_version=verifier_version,
            build_config=build_config,
        )
        return cls(
            store=store,
            envelope=ArtifactEnvelope(content=content, stamp=stamp, support=manifest),
            documents=tuple(documents),
            disabled_records=disabled,
            compatibility_policy=ArtifactCompatibilityPolicy.exact_for_stamp(stamp),
        )

    @property
    def content_hash(self) -> str:
        return self.envelope.stamp.content_hash

    def search(
        self,
        query: str,
        *,
        query_id: str,
        valid_at: str,
        top_k: int = 5,
        metadata_filter: dict[str, str] | None = None,
    ) -> RetrievalResult:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        metadata_filter = {
            str(key): str(value) for key, value in (metadata_filter or {}).items()
        }
        # The global stamp is checked, while per-document support is evaluated
        # below.  That lets a tombstone disable its document immediately without
        # making an unrelated part of a large index unavailable.
        self.envelope.validate(
            self.store,
            compatibility_policy=self.compatibility_policy,
            valid_at=valid_at,
            check_support=False,
        )
        snapshot = resolve(self.store, valid_at=valid_at)
        usable = snapshot.usable_event_ids()
        resolved_by_id = {record.event_id: record for record in snapshot.records}
        query_tokens = Counter(_tokens(query))
        query_features = _features(query)
        candidates = []
        for document in self.documents:
            reasons = []
            if document.event_id not in usable:
                reasons.append("support_inactive")
            for key, expected in metadata_filter.items():
                if document.metadata.get(key) != expected:
                    reasons.append(f"metadata:{key}")
            doc_tokens = Counter(_tokens(document.text))
            overlap = sum(
                min(count, doc_tokens.get(token, 0))
                for token, count in query_tokens.items()
            )
            lexical = overlap / max(1, sum(query_tokens.values()))
            semantic = _cosine(query_features, _features(document.text))
            phrase = 1.0 if query.lower().strip() in document.text.lower() else 0.0
            reranker = 0.45 * lexical + 0.45 * semantic + 0.10 * phrase
            final = reranker if not reasons else 0.0
            candidates.append(
                {
                    "event_id": document.event_id,
                    "supporting_event_ids": list(document.support.supporting_event_ids),
                    "control_event_ids": list(
                        resolved_by_id.get(document.event_id).control_event_ids
                        if resolved_by_id.get(document.event_id)
                        else ()
                    ),
                    "resolved_status": (
                        resolved_by_id[document.event_id].status
                        if document.event_id in resolved_by_id
                        else "missing"
                    ),
                    "filter_passed": not reasons,
                    "filter_reasons": reasons,
                    "lexical_score": round(lexical, 12),
                    "embedding_score": round(semantic, 12),
                    "reranker_score": round(reranker, 12),
                    "final_score": round(final, 12),
                    "selected": False,
                }
            )
        ranked = sorted(
            (item for item in candidates if item["filter_passed"] and item["final_score"] > 0),
            key=lambda item: (-item["final_score"], item["event_id"]),
        )
        selected = ranked[:top_k]
        selected_ids = {item["event_id"] for item in selected}
        for candidate in candidates:
            candidate["selected"] = candidate["event_id"] in selected_ids
        if not selected:
            margin = None
        elif len(ranked) > len(selected):
            margin = round(
                selected[-1]["final_score"] - ranked[len(selected)]["final_score"], 12
            )
        else:
            margin = round(selected[-1]["final_score"], 12)
        record = {
            "query_id": query_id,
            "query": query,
            "query_version": QUERY_VERSION,
            "valid_at": valid_at,
            "filters": dict(sorted(metadata_filter.items())),
            "index_artifact_id": self.envelope.stamp.artifact_id,
            "index_content_hash": self.content_hash,
            "index_frontier": {
                "sequence": self.envelope.stamp.ledger_frontier_sequence,
                "chain_hash": self.envelope.stamp.ledger_frontier_hash,
            },
            "decision_frontier": snapshot.frontier.to_dict(),
            "component_versions": self.envelope.stamp.component_versions,
            "candidates": sorted(candidates, key=lambda item: item["event_id"]),
            "selected_event_ids": [item["event_id"] for item in selected],
            "top_k": top_k,
            "top_k_boundary_margin": margin,
        }
        journal_hash = self.store.append_retrieval_record(record)
        by_id = {document.event_id: document for document in self.documents}
        hits = tuple(
            RetrievalHit(
                event_id=item["event_id"],
                text=by_id[item["event_id"]].text,
                region=by_id[item["event_id"]].region,
                score=item["final_score"],
                support=by_id[item["event_id"]].support,
            )
            for item in selected
        )
        return RetrievalResult(
            query_id=query_id,
            hits=hits,
            journal_hash=journal_hash,
            boundary_margin=margin,
        )

    def selection_manifest(self, query_id: str) -> SupportManifest:
        record = next(
            (
                item
                for item in self.store.retrieval_records()
                if item["query_id"] == query_id
            ),
            None,
        )
        if record is None:
            raise KeyError(query_id)
        return SupportManifest.create(
            supporting_event_ids=record["selected_event_ids"],
            contradicting_event_ids=[
                control_id
                for item in record["candidates"]
                if item["resolved_status"] in {"refuted", "retracted", "tombstone"}
                for control_id in item["control_event_ids"]
            ],
            candidate_event_ids=[
                event_id
                for item in record["candidates"]
                for event_id in (
                    *item["supporting_event_ids"],
                    *item["control_event_ids"],
                )
            ],
            require_all_support=True,
            minimum_live_support=1 if record["selected_event_ids"] else 0,
        )

    def document_ids(self) -> set[str]:
        return {document.event_id for document in self.documents}

    def audit_hash(self) -> str:
        return sha256_json(self.envelope.content)
