"""SQLite-backed append-only authority and append-only selection journal."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json, sha256_json
from .events import Event

GENESIS_HASH = "0" * 64


class EventConflictError(RuntimeError):
    pass


class SimulatedWriteInterruption(RuntimeError):
    """Test seam used to prove transaction recovery at every write boundary."""


@dataclass(frozen=True)
class LedgerFrontier:
    sequence: int
    chain_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {"sequence": self.sequence, "chain_hash": self.chain_hash}


class AppendOnlyEventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_hash TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    chain_hash TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS retrieval_journal (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT NOT NULL UNIQUE,
                    record_hash TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS events_no_update
                BEFORE UPDATE ON events BEGIN
                    SELECT RAISE(ABORT, 'events are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS events_no_delete
                BEFORE DELETE ON events BEGIN
                    SELECT RAISE(ABORT, 'events are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS journal_no_update
                BEFORE UPDATE ON retrieval_journal BEGIN
                    SELECT RAISE(ABORT, 'retrieval journal is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS journal_no_delete
                BEFORE DELETE ON retrieval_journal BEGIN
                    SELECT RAISE(ABORT, 'retrieval journal is append-only');
                END;
                """
            )
            connection.commit()

    def append(self, event: Event) -> LedgerFrontier:
        return self.append_batch([event])

    def append_batch(
        self, events: Iterable[Event], *, interrupt_after: int | None = None
    ) -> LedgerFrontier:
        """Atomically append a validated batch.

        ``interrupt_after`` exists only for recovery tests.  Zero interrupts
        before the first insert; N interrupts after N inserts but before commit.
        SQLite must roll the entire batch back in every case.
        """

        batch = list(events)
        if not batch:
            return self.frontier()
        for event in batch:
            event.validate()
        ids = [event.event_id for event in batch]
        if len(ids) != len(set(ids)):
            raise EventConflictError("a batch cannot contain duplicate event IDs")
        batch_ids = set(ids)

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in ids)
            existing_rows = connection.execute(
                f"SELECT event_id, event_json FROM events WHERE event_id IN ({placeholders})",
                ids,
            ).fetchall()
            existing = {row["event_id"]: row["event_json"] for row in existing_rows}
            for event in batch:
                if event.event_id in existing and existing[event.event_id] != event.canonical:
                    raise EventConflictError(
                        f"event ID {event.event_id!r} already has different content"
                    )

            references = {
                reference
                for event in batch
                for reference in (*event.parent_ids, *event.target_event_ids)
                if reference not in batch_ids
            }
            if references:
                placeholders = ",".join("?" for _ in references)
                found = {
                    row[0]
                    for row in connection.execute(
                        f"SELECT event_id FROM events WHERE event_id IN ({placeholders})",
                        sorted(references),
                    )
                }
                missing = sorted(references - found)
                if missing:
                    raise ValueError(f"event references are missing: {missing}")

            if interrupt_after == 0:
                raise SimulatedWriteInterruption("interrupted before first insert")
            row = connection.execute(
                "SELECT chain_hash FROM events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            chain = row[0] if row else GENESIS_HASH
            inserted = 0
            for event in batch:
                if event.event_id in existing:
                    continue
                chain = hashlib.sha256(
                    f"{chain}:{event.event_hash}".encode("utf-8")
                ).hexdigest()
                connection.execute(
                    "INSERT INTO events(event_id, event_hash, event_json, chain_hash) "
                    "VALUES (?, ?, ?, ?)",
                    (event.event_id, event.event_hash, event.canonical, chain),
                )
                inserted += 1
                if interrupt_after is not None and inserted == interrupt_after:
                    raise SimulatedWriteInterruption(
                        f"interrupted after insert boundary {inserted}"
                    )
            connection.commit()
            return self.frontier(connection)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def events(self, *, through_sequence: int | None = None) -> list[Event]:
        sql = "SELECT event_json FROM events"
        params: tuple[Any, ...] = ()
        if through_sequence is not None:
            sql += " WHERE sequence <= ?"
            params = (through_sequence,)
        sql += " ORDER BY sequence"
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [Event.from_dict(json.loads(row[0])) for row in rows]

    def get_event(self, event_id: str) -> Event | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT event_json FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return Event.from_dict(json.loads(row[0])) if row else None

    def event_sequence(self, event_id: str) -> int | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT sequence FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return int(row[0]) if row else None

    def count(self) -> int:
        return self.frontier().sequence

    def frontier(self, connection: sqlite3.Connection | None = None) -> LedgerFrontier:
        owns_connection = connection is None
        connection = connection or self._connect()
        try:
            row = connection.execute(
                "SELECT sequence, chain_hash FROM events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            return (
                LedgerFrontier(int(row[0]), str(row[1]))
                if row
                else LedgerFrontier(0, GENESIS_HASH)
            )
        finally:
            if owns_connection:
                connection.close()

    def frontier_at(self, sequence: int) -> LedgerFrontier | None:
        if sequence == 0:
            return LedgerFrontier(0, GENESIS_HASH)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT sequence, chain_hash FROM events WHERE sequence = ?", (sequence,)
            ).fetchone()
        return LedgerFrontier(int(row[0]), str(row[1])) if row else None

    def verify_integrity(self) -> None:
        chain = GENESIS_HASH
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT event_hash, event_json, chain_hash FROM events ORDER BY sequence"
            ).fetchall()
        for row in rows:
            event = Event.from_dict(json.loads(row["event_json"]))
            if event.event_hash != row["event_hash"]:
                raise EventConflictError("stored event hash does not match event content")
            chain = hashlib.sha256(
                f"{chain}:{event.event_hash}".encode("utf-8")
            ).hexdigest()
            if chain != row["chain_hash"]:
                raise EventConflictError("ledger chain hash mismatch")

    def append_retrieval_record(self, record: dict[str, Any]) -> str:
        query_id = str(record.get("query_id", ""))
        if not query_id:
            raise ValueError("retrieval record requires query_id")
        canonical = canonical_json(record)
        record_hash = sha256_json(record)
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT record_json FROM retrieval_journal WHERE query_id = ?", (query_id,)
            ).fetchone()
            if existing:
                if existing[0] != canonical:
                    raise EventConflictError(
                        f"query ID {query_id!r} already has a different journal record"
                    )
                return record_hash
            connection.execute(
                "INSERT INTO retrieval_journal(query_id, record_hash, record_json) "
                "VALUES (?, ?, ?)",
                (query_id, record_hash, canonical),
            )
            connection.commit()
        return record_hash

    def retrieval_records(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT record_json FROM retrieval_journal ORDER BY sequence"
            ).fetchall()
        return [json.loads(row[0]) for row in rows]
