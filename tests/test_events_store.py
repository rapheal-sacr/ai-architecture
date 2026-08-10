from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

from wamrx.events import Event, SpeechAct
from wamrx.resolver import replay, resolve
from wamrx.store import (
    AppendOnlyEventStore,
    EventConflictError,
    SimulatedWriteInterruption,
)

NOW = "2026-08-09T10:00:00+00:00"
VALID_AT = "2026-08-09T12:00:00+00:00"


def fact(event_id: str, text: str = "a fact", *, valid_to: str | None = None) -> Event:
    return Event.create(
        event_id=event_id,
        transaction_time=NOW,
        valid_from="2026-08-09T00:00:00+00:00",
        valid_to=valid_to,
        actor="test",
        source_id="test-source",
        modality="text",
        speech_act=SpeechAct.OBSERVED,
        payload={"text": text, "region": "test"},
        verifier_class="executable",
        verifier_id="test-verifier",
    )


def control(event_id: str, act: SpeechAct, target: str) -> Event:
    return Event.create(
        event_id=event_id,
        transaction_time="2026-08-09T11:00:00+00:00",
        valid_from="2026-08-09T00:00:00+00:00",
        actor="test",
        source_id="test-verifier",
        modality="structured-control",
        speech_act=act,
        payload={"reason": act.value},
        target_event_ids=(target,),
        parent_ids=(target,),
        verifier_class="executable",
        verifier_id="test-verifier",
    )


class EventContractTests(unittest.TestCase):
    def test_requires_timezone_and_control_target(self) -> None:
        with self.assertRaises(ValueError):
            Event.create(
                event_id="bad-time",
                transaction_time="2026-08-09T10:00:00",
                valid_from=NOW,
                actor="test",
                source_id="source",
                modality="text",
                speech_act=SpeechAct.OBSERVED,
            )
        with self.assertRaises(ValueError):
            Event.create(
                event_id="bad-control",
                transaction_time=NOW,
                valid_from=NOW,
                actor="test",
                source_id="source",
                modality="structured-control",
                speech_act=SpeechAct.TOMBSTONE,
            )
        with self.assertRaises(ValueError):
            Event.create(
                event_id="missing-verifier-id",
                transaction_time=NOW,
                valid_from=NOW,
                actor="test",
                source_id="source",
                modality="text",
                speech_act=SpeechAct.OBSERVED,
                verifier_class="grounded",
            )

    def test_payload_hash_is_enforced(self) -> None:
        event = fact("fact")
        with self.assertRaises(ValueError):
            dataclasses.replace(event, payload={"text": "tampered"}).validate()


class StoreAndReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "ledger.sqlite"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_atomic_batch_at_every_boundary(self) -> None:
        batch = [fact("a"), fact("b"), fact("c")]
        for boundary in range(4):
            store = AppendOnlyEventStore(Path(self.temp.name) / f"b{boundary}.sqlite")
            with self.assertRaises(SimulatedWriteInterruption):
                store.append_batch(batch, interrupt_after=boundary)
            self.assertEqual(store.count(), 0)
            store.verify_integrity()

    def test_idempotence_conflict_and_missing_reference(self) -> None:
        store = AppendOnlyEventStore(self.path)
        original = fact("a")
        store.append(original)
        store.append(original)
        self.assertEqual(store.count(), 1)
        with self.assertRaises(EventConflictError):
            store.append(fact("a", "different"))
        with self.assertRaises(ValueError):
            store.append(control("delete-missing", SpeechAct.TOMBSTONE, "missing"))
        self.assertEqual(store.count(), 1)

    def test_correction_and_contradiction_are_deterministic(self) -> None:
        old = fact("old", "office montreal")
        new = fact("new", "office toronto")
        refute = control("refute-old", SpeechAct.REFUTED, "old")
        verify = control("verify-new", SpeechAct.VERIFIED, "new")
        events = [old, new, refute, verify]
        store = AppendOnlyEventStore(self.path)
        store.append_batch(events)
        first = resolve(store, valid_at=VALID_AT)
        self.assertEqual(first.record("old").status, "refuted")
        self.assertEqual(first.record("new").status, "verified")
        reversed_hash = replay(reversed(events), valid_at=VALID_AT).snapshot_hash
        self.assertEqual(first.snapshot_hash, reversed_hash)
        self.assertEqual(
            len({resolve(store, valid_at=VALID_AT).snapshot_hash for _ in range(10)}),
            1,
        )

    def test_valid_time_is_explicit(self) -> None:
        event = fact("expired", valid_to="2026-08-09T11:00:00+00:00")
        store = AppendOnlyEventStore(self.path)
        store.append(event)
        record = resolve(store, valid_at=VALID_AT).record("expired")
        self.assertEqual(record.status, "out_of_validity")


if __name__ == "__main__":
    unittest.main()
