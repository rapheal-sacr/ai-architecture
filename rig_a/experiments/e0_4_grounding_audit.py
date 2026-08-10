"""E0.4 -- I1 claim-level grounding and auditability.

CLAIM
    Every promoted capability resolves through typed provenance to at least one
    observed event, and every non-unverified event names a verifier and witness.

KILL
    Any accepted promotion without an observed root; grounding coverage below
    1.0; a missing/untyped witness accepted; or rejection exposing a partial
    batch.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wamrx.contracts import load_contracts  # noqa: E402
from wamrx.events import Event, SpeechAct  # noqa: E402
from wamrx.grounding import GroundingAuditor, GroundingError, is_promotion  # noqa: E402
from wamrx.store import AppendOnlyEventStore  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def make(
    event_id: str,
    act: SpeechAct,
    minute: int,
    *,
    parents: tuple[str, ...] = (),
    witnesses: tuple[str, ...] = (),
    promotion: bool = False,
    verifier_class: str = "grounded",
) -> Event:
    payload = {"text": event_id, "region": "audit"}
    if promotion:
        payload["promotion"] = True
    return Event.create(
        event_id=event_id,
        transaction_time=f"2026-08-10T12:{minute:02d}:00+00:00",
        valid_from="2026-08-10T00:00:00+00:00",
        actor="audit-world",
        source_id="source:audit-world",
        verifier_id=("verifier:audit" if verifier_class != "unverified" else None),
        modality="structured-text",
        speech_act=act,
        payload=payload,
        parent_ids=parents,
        verifier_class=verifier_class,
        provenance_witnesses=witnesses,
    )


def run() -> dict:
    contracts = load_contracts(ROOT / "contracts" / "wamrx_milestone2_foundation.json")
    with tempfile.TemporaryDirectory(prefix="wamrx-e0-4-") as directory:
        store = AppendOnlyEventStore(pathlib.Path(directory) / "audit.sqlite")
        events = [
            make(
                "observed-a",
                SpeechAct.OBSERVED,
                0,
                witnesses=("external:human:decision-a",),
            ),
            make(
                "asserted-a",
                SpeechAct.ASSERTED,
                1,
                parents=("observed-a",),
                witnesses=("event:observed-a",),
            ),
            make(
                "promotion-a",
                SpeechAct.INFERRED,
                2,
                parents=("asserted-a",),
                witnesses=("event:asserted-a",),
                promotion=True,
                verifier_class="executable",
            ),
            make(
                "observed-b",
                SpeechAct.OBSERVED,
                3,
                witnesses=("external:test:result-b",),
                verifier_class="executable",
            ),
            make(
                "promotion-b",
                SpeechAct.INFERRED,
                4,
                parents=("observed-b",),
                witnesses=("event:observed-b",),
                promotion=True,
                verifier_class="executable",
            ),
        ]
        store.append_batch(events)
        auditor = GroundingAuditor(store.events())
        promotions = [event for event in store.events() if is_promotion(event)]
        reports = [auditor.report(event.event_id) for event in promotions]
        grounding_coverage = (
            sum(report.grounded for report in reports) / len(reports)
            if reports
            else 0.0
        )

        typed_and_identified = all(
            event.verifier_class == "unverified"
            or (
                event.verifier_id
                and event.provenance_witnesses
                and all(
                    witness.startswith(("event:", "external:"))
                    for witness in event.provenance_witnesses
                )
            )
            for event in store.events()
        )

        before = store.count()
        ungrounded_rejected = False
        try:
            store.append_batch(
                [
                    make(
                        "unverified-synthetic",
                        SpeechAct.INFERRED,
                        10,
                        witnesses=(),
                        verifier_class="unverified",
                    ),
                    make(
                        "promotion-ungrounded",
                        SpeechAct.INFERRED,
                        11,
                        parents=("unverified-synthetic",),
                        witnesses=("event:unverified-synthetic",),
                        promotion=True,
                        verifier_class="executable",
                    ),
                ]
            )
        except GroundingError:
            ungrounded_rejected = True

        untyped_rejected = False
        try:
            make(
                "untyped-witness",
                SpeechAct.ASSERTED,
                12,
                witnesses=("observed-a",),
            )
        except ValueError:
            untyped_rejected = True

        checks = {
            "I1_promoted_grounding_coverage": grounding_coverage == 1.0,
            "I1_typed_witnesses_and_verifier_identity": typed_and_identified,
            "I1_expected_observed_roots": (
                reports[0].observed_root_ids == ("observed-a",)
                and reports[1].observed_root_ids == ("observed-b",)
            ),
            "I1_ungrounded_promotion_rejected": ungrounded_rejected,
            "I1_untyped_witness_rejected": untyped_rejected,
            "I1_rejection_is_atomic": store.count() == before,
        }
        return {
            "experiment": "E0.4",
            "claim": "promoted capabilities have complete typed grounding to observations",
            "contract_count": len(contracts),
            "promoted_capability_count": len(promotions),
            "grounding_coverage": grounding_coverage,
            "reports": [
                {
                    "event_id": report.event_id,
                    "closure": list(report.closure_event_ids),
                    "observed_roots": list(report.observed_root_ids),
                }
                for report in reports
            ],
            "checks": checks,
            "verdict": "PASS" if all(checks.values()) else "FAIL",
        }


def main() -> int:
    result = run()
    path = ROOT / "results" / "e0_4_grounding_audit.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
