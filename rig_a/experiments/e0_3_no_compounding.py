"""E0.3 -- I3 no-compounding under synthetic and inferred experience.

CLAIM
    A durable promotion may descend through inferred and synthetic events, but
    its transitive parent/witness closure must contain an observed ledger event.

REGISTERED KILL CRITERIA (M2-grounding-v1)
    G1 a missing event:<id> witness is accepted;
    G2 an inferred/synthetic-only promotion is accepted;
    G3 a grounded multi-hop promotion is rejected;
    G4 an external witness on an inferred event is counted as observed;
    G5 rejection exposes any row from the failed atomic batch.
    G6 an inferred/external witness creates an extra observed oracle root.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from wamrx.contracts import load_contracts  # noqa: E402
from wamrx.events import Event, SpeechAct  # noqa: E402
from wamrx.grounding import GroundingAuditor, GroundingError  # noqa: E402
from wamrx.store import AppendOnlyEventStore  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
TIME = "2026-08-10T12:00:00+00:00"


def event(
    event_id: str,
    speech_act: SpeechAct,
    minute: int,
    *,
    parents: tuple[str, ...] = (),
    witnesses: tuple[str, ...] = (),
    promotion: bool = False,
    synthetic: bool = False,
    verifier_class: str = "unverified",
) -> Event:
    payload = {
        "text": event_id,
        "region": "grounding",
        "synthetic": synthetic,
    }
    if promotion:
        payload["promotion"] = {"capability": event_id}
    return Event.create(
        event_id=event_id,
        transaction_time=f"2026-08-10T11:{minute:02d}:00+00:00",
        valid_from="2026-08-10T00:00:00+00:00",
        actor="grounding-world",
        source_id="source:grounding-world",
        verifier_id=(
            "verifier:grounding-world" if verifier_class != "unverified" else None
        ),
        modality="structured-text",
        speech_act=speech_act,
        payload=payload,
        parent_ids=parents,
        verifier_class=verifier_class,
        provenance_witnesses=witnesses,
    )


def run() -> dict:
    contracts = load_contracts(ROOT / "contracts" / "wamrx_milestone2_foundation.json")
    with tempfile.TemporaryDirectory(prefix="wamrx-e0-3-") as directory:
        store = AppendOnlyEventStore(pathlib.Path(directory) / "grounding.sqlite")
        grounded_batch = [
            event(
                "observation-root",
                SpeechAct.OBSERVED,
                0,
                witnesses=("external:sensor:observation-root",),
                verifier_class="grounded",
            ),
            event(
                "inferred-gap",
                SpeechAct.INFERRED,
                1,
                parents=("observation-root",),
                witnesses=("event:observation-root",),
                verifier_class="grounded",
            ),
            event(
                "synthetic-task",
                SpeechAct.INFERRED,
                2,
                parents=("inferred-gap",),
                witnesses=("event:inferred-gap",),
                synthetic=True,
                verifier_class="grounded",
            ),
            event(
                "grounded-promotion",
                SpeechAct.INFERRED,
                3,
                parents=("synthetic-task",),
                witnesses=("event:synthetic-task",),
                promotion=True,
                synthetic=True,
                verifier_class="executable",
            ),
        ]
        store.append_batch(grounded_batch)
        grounded_report = GroundingAuditor(store.events()).report(
            "grounded-promotion"
        )

        before_bad = store.count()
        bad_batch = [
            event(
                "synthetic-root",
                SpeechAct.INFERRED,
                10,
                synthetic=True,
            ),
            event(
                "synthetic-only-promotion",
                SpeechAct.INFERRED,
                11,
                parents=("synthetic-root",),
                witnesses=("event:synthetic-root",),
                promotion=True,
                synthetic=True,
                verifier_class="executable",
            ),
        ]
        synthetic_only_rejected = False
        try:
            store.append_batch(bad_batch)
        except GroundingError:
            synthetic_only_rejected = True
        bad_batch_rows = store.count() - before_bad

        external_inference_batch = [
            event(
                "external-inference",
                SpeechAct.INFERRED,
                12,
                witnesses=("external:model:self-report",),
                synthetic=True,
                verifier_class="grounded",
            ),
            event(
                "external-inference-promotion",
                SpeechAct.INFERRED,
                13,
                parents=("external-inference",),
                witnesses=("event:external-inference",),
                promotion=True,
                synthetic=True,
                verifier_class="executable",
            ),
        ]
        external_inference_rejected = False
        try:
            store.append_batch(external_inference_batch)
        except GroundingError:
            external_inference_rejected = True

        missing_witness_rejected = False
        try:
            store.append(
                event(
                    "missing-witness-event",
                    SpeechAct.INFERRED,
                    14,
                    witnesses=("event:not-in-ledger",),
                    verifier_class="grounded",
                )
            )
        except ValueError:
            missing_witness_rejected = True

        checks = {
            "G1_missing_ledger_witness_rejected": missing_witness_rejected,
            "G2_synthetic_only_promotion_rejected": synthetic_only_rejected,
            "G3_grounded_multihop_promotion_accepted": (
                store.get_event("grounded-promotion") is not None
                and grounded_report.grounded
                and grounded_report.observed_root_ids == ("observation-root",)
            ),
            "G4_external_inference_not_observed": external_inference_rejected,
            "G5_rejected_batch_is_atomic": bad_batch_rows == 0
            and store.get_event("synthetic-root") is None,
            "G6_oracle_roots_conserved": (
                grounded_report.observed_root_ids == ("observation-root",)
                and grounded_report.external_witness_ids
                == ("external:sensor:observation-root",)
            ),
        }
        return {
            "experiment": "E0.3",
            "claim": "synthetic and inferred chains cannot compound into ungrounded promotion",
            "contract_count": len(contracts),
            "grounded_promotion_closure": list(
                grounded_report.closure_event_ids
            ),
            "observed_roots": list(grounded_report.observed_root_ids),
            "checks": checks,
            "verdict": "PASS" if all(checks.values()) else "FAIL",
        }


def main() -> int:
    result = run()
    path = ROOT / "results" / "e0_3_no_compounding.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
