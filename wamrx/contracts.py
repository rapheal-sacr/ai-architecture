"""Fail-closed architecture and experiment contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class MechanismDeclaration:
    mechanism_id: str
    name: str
    authoritative_inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    assumptions: tuple[str, ...]
    scope: str
    cost: str
    tradeoff: str
    failure_response: str
    verifier_tier: str
    permitted_mutations: tuple[str, ...]
    registered_thresholds: dict[str, float | int | str]
    kill_criteria: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MechanismDeclaration":
        declaration = cls(
            mechanism_id=str(value.get("mechanism_id", "")),
            name=str(value.get("name", "")),
            authoritative_inputs=tuple(value.get("authoritative_inputs", ())),
            outputs=tuple(value.get("outputs", ())),
            assumptions=tuple(value.get("assumptions", ())),
            scope=str(value.get("scope", "")),
            cost=str(value.get("cost", "")),
            tradeoff=str(value.get("tradeoff", "")),
            failure_response=str(value.get("failure_response", "")),
            verifier_tier=str(value.get("verifier_tier", "")),
            permitted_mutations=tuple(value.get("permitted_mutations", ())),
            registered_thresholds=dict(value.get("registered_thresholds", {})),
            kill_criteria=tuple(value.get("kill_criteria", ())),
        )
        declaration.validate()
        return declaration

    def validate(self) -> None:
        scalars = {
            "mechanism_id": self.mechanism_id,
            "name": self.name,
            "scope": self.scope,
            "cost": self.cost,
            "tradeoff": self.tradeoff,
            "failure_response": self.failure_response,
            "verifier_tier": self.verifier_tier,
        }
        empty = [name for name, value in scalars.items() if not value.strip()]
        collections = {
            "authoritative_inputs": self.authoritative_inputs,
            "outputs": self.outputs,
            "assumptions": self.assumptions,
            "permitted_mutations": self.permitted_mutations,
            "registered_thresholds": self.registered_thresholds,
            "kill_criteria": self.kill_criteria,
        }
        empty.extend(name for name, value in collections.items() if not value)
        if empty:
            raise ContractError(
                f"{self.mechanism_id or '<unnamed>'} has ambiguous/missing contract fields: "
                + ", ".join(empty)
            )
        if self.verifier_tier not in {"executable", "grounded", "judge", "sealed"}:
            raise ContractError(
                f"{self.mechanism_id}: unknown verifier tier {self.verifier_tier!r}"
            )
        for name, threshold in self.registered_thresholds.items():
            if not name or threshold in (None, ""):
                raise ContractError(
                    f"{self.mechanism_id}: thresholds must have names and fixed values"
                )
        if any(not criterion.strip() for criterion in self.kill_criteria):
            raise ContractError(f"{self.mechanism_id}: kill criteria cannot be blank")


def load_contracts(path: str | Path) -> tuple[MechanismDeclaration, ...]:
    value = json.loads(Path(path).read_text())
    if value.get("contract_schema_version") != 1:
        raise ContractError("unsupported contract schema version")
    declarations = tuple(
        MechanismDeclaration.from_dict(item) for item in value.get("mechanisms", ())
    )
    if not declarations:
        raise ContractError("contract registry has no mechanisms")
    ids = [item.mechanism_id for item in declarations]
    if len(ids) != len(set(ids)):
        raise ContractError("mechanism IDs must be unique")
    return declarations
