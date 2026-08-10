"""Common lineage contract for every derived artifact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import sha256_json
from .resolver import resolve
from .store import AppendOnlyEventStore

ARTIFACT_SCHEMA_VERSION = 1


class InvalidArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactCompatibilityPolicy:
    """Directional compatibility from a stored artifact into an active runtime."""

    active_base_weight_version: str
    active_component_versions: dict[str, str]
    active_ontology_version: str
    active_verifier_version: str
    compatible_base_weight_predecessors: tuple[str, ...] = ()
    compatible_component_predecessors: dict[str, tuple[str, ...]] | None = None
    compatible_ontology_predecessors: tuple[str, ...] = ()
    compatible_verifier_predecessors: tuple[str, ...] = ()

    @classmethod
    def exact_for_stamp(cls, stamp: "ArtifactStamp") -> "ArtifactCompatibilityPolicy":
        return cls(
            active_base_weight_version=stamp.base_weight_version,
            active_component_versions=dict(stamp.component_versions),
            active_ontology_version=stamp.ontology_version,
            active_verifier_version=stamp.verifier_version,
        )

    def validate(self, stamp: "ArtifactStamp") -> None:
        if stamp.base_weight_version not in {
            self.active_base_weight_version,
            *self.compatible_base_weight_predecessors,
        }:
            raise InvalidArtifactError(
                "artifact base-weight version is incompatible with the active runtime"
            )
        if set(stamp.component_versions) != set(self.active_component_versions):
            raise InvalidArtifactError(
                "artifact component set is incompatible with the active runtime"
            )
        predecessors = self.compatible_component_predecessors or {}
        for component, active_version in self.active_component_versions.items():
            allowed = {active_version, *predecessors.get(component, ())}
            if stamp.component_versions[component] not in allowed:
                raise InvalidArtifactError(
                    f"artifact component {component!r} is incompatible: "
                    f"stored={stamp.component_versions[component]!r}, "
                    f"active={active_version!r}"
                )
        if stamp.ontology_version not in {
            self.active_ontology_version,
            *self.compatible_ontology_predecessors,
        }:
            raise InvalidArtifactError(
                "artifact ontology version has no registered migration into the active runtime"
            )
        if stamp.verifier_version not in {
            self.active_verifier_version,
            *self.compatible_verifier_predecessors,
        }:
            raise InvalidArtifactError(
                "artifact verifier version is incompatible with the active runtime"
            )


@dataclass(frozen=True)
class SupportManifest:
    supporting_event_ids: tuple[str, ...]
    contradicting_event_ids: tuple[str, ...]
    candidate_event_ids: tuple[str, ...]
    require_all_support: bool = True
    minimum_live_support: int = 1

    @classmethod
    def create(
        cls,
        *,
        supporting_event_ids: list[str] | tuple[str, ...],
        contradicting_event_ids: list[str] | tuple[str, ...] = (),
        candidate_event_ids: list[str] | tuple[str, ...] = (),
        require_all_support: bool = True,
        minimum_live_support: int = 1,
    ) -> "SupportManifest":
        supporting = tuple(sorted(set(supporting_event_ids)))
        contradicting = tuple(sorted(set(contradicting_event_ids)))
        candidates = tuple(sorted(set(candidate_event_ids)))
        if not candidates:
            candidates = tuple(sorted(set((*supporting, *contradicting))))
        manifest = cls(
            supporting_event_ids=supporting,
            contradicting_event_ids=contradicting,
            candidate_event_ids=candidates,
            require_all_support=require_all_support,
            minimum_live_support=minimum_live_support,
        )
        manifest.validate_shape()
        return manifest

    def validate_shape(self) -> None:
        if self.minimum_live_support < 0:
            raise InvalidArtifactError("minimum_live_support cannot be negative")
        if self.minimum_live_support > len(self.supporting_event_ids):
            raise InvalidArtifactError("minimum live support exceeds recorded support")
        if set(self.supporting_event_ids) & set(self.contradicting_event_ids):
            raise InvalidArtifactError("supporting and contradicting evidence overlap")
        recorded = set(self.supporting_event_ids) | set(self.contradicting_event_ids)
        if not recorded <= set(self.candidate_event_ids):
            raise InvalidArtifactError(
                "candidate set must contain every supporting and contradicting event"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "supporting_event_ids": list(self.supporting_event_ids),
            "contradicting_event_ids": list(self.contradicting_event_ids),
            "candidate_event_ids": list(self.candidate_event_ids),
            "require_all_support": self.require_all_support,
            "minimum_live_support": self.minimum_live_support,
        }


@dataclass(frozen=True)
class ArtifactStamp:
    artifact_id: str
    artifact_type: str
    ledger_frontier_sequence: int
    ledger_frontier_hash: str
    base_weight_version: str
    component_versions: dict[str, str]
    ontology_version: str
    verifier_version: str
    build_config_hash: str
    content_hash: str
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        artifact_id: str,
        artifact_type: str,
        content: Any,
        store: AppendOnlyEventStore,
        base_weight_version: str,
        component_versions: dict[str, str],
        ontology_version: str,
        verifier_version: str,
        build_config: dict[str, Any],
    ) -> "ArtifactStamp":
        frontier = store.frontier()
        stamp = cls(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            ledger_frontier_sequence=frontier.sequence,
            ledger_frontier_hash=frontier.chain_hash,
            base_weight_version=base_weight_version,
            component_versions=dict(sorted(component_versions.items())),
            ontology_version=ontology_version,
            verifier_version=verifier_version,
            build_config_hash=sha256_json(build_config),
            content_hash=sha256_json(content),
        )
        stamp.validate_shape()
        return stamp

    def validate_shape(self) -> None:
        required = {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "ledger_frontier_hash": self.ledger_frontier_hash,
            "base_weight_version": self.base_weight_version,
            "ontology_version": self.ontology_version,
            "verifier_version": self.verifier_version,
            "build_config_hash": self.build_config_hash,
            "content_hash": self.content_hash,
        }
        missing = [name for name, value in required.items() if not value]
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            missing.append("supported schema_version")
        if self.ledger_frontier_sequence < 0:
            missing.append("nonnegative ledger_frontier_sequence")
        if not self.component_versions or any(
            not key or not value for key, value in self.component_versions.items()
        ):
            missing.append("component_versions")
        if missing:
            raise InvalidArtifactError(
                "artifact stamp is incomplete: " + ", ".join(missing)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "ledger_frontier_sequence": self.ledger_frontier_sequence,
            "ledger_frontier_hash": self.ledger_frontier_hash,
            "base_weight_version": self.base_weight_version,
            "component_versions": self.component_versions,
            "ontology_version": self.ontology_version,
            "verifier_version": self.verifier_version,
            "build_config_hash": self.build_config_hash,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class ArtifactEnvelope:
    content: Any
    stamp: ArtifactStamp
    support: SupportManifest

    def validate(
        self,
        store: AppendOnlyEventStore,
        *,
        compatibility_policy: ArtifactCompatibilityPolicy | None,
        valid_at: str,
        check_support: bool = True,
    ) -> None:
        self.stamp.validate_shape()
        if compatibility_policy is None:
            raise InvalidArtifactError(
                "artifact reads require an explicit active runtime compatibility policy"
            )
        compatibility_policy.validate(self.stamp)
        self.support.validate_shape()
        if sha256_json(self.content) != self.stamp.content_hash:
            raise InvalidArtifactError("artifact content hash mismatch")
        frontier = store.frontier_at(self.stamp.ledger_frontier_sequence)
        if frontier is None or frontier.chain_hash != self.stamp.ledger_frontier_hash:
            raise InvalidArtifactError("artifact ledger frontier is missing or mismatched")

        lineage_ids = set(self.support.candidate_event_ids)
        missing = sorted(
            event_id
            for event_id in lineage_ids
            if (sequence := store.event_sequence(event_id)) is None
            or sequence > self.stamp.ledger_frontier_sequence
        )
        if missing:
            raise InvalidArtifactError(
                f"artifact lineage is absent from its frontier: {missing}"
            )
        if not check_support:
            return
        snapshot = resolve(store, valid_at=valid_at)
        usable = snapshot.usable_event_ids()
        live = set(self.support.supporting_event_ids) & usable
        if self.support.require_all_support and live != set(
            self.support.supporting_event_ids
        ):
            invalid = sorted(set(self.support.supporting_event_ids) - live)
            raise InvalidArtifactError(f"artifact support is no longer usable: {invalid}")
        if len(live) < self.support.minimum_live_support:
            raise InvalidArtifactError(
                "artifact has insufficient surviving support: "
                f"{len(live)} < {self.support.minimum_live_support}"
            )
