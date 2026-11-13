"""Cryptographic SHA-256 hash-chained append-only audit ledger (ADR-005)."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

GENESIS_PREV_HASH = "0" * 64


class AuditEntry(BaseModel):
    sequence_num: int
    timestamp: float
    batch_id: str
    prev_hash: str
    payload: dict[str, Any] = Field(default_factory=dict)
    entry_hash: str


def compute_canonical_hash(prev_hash: str, payload: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of prev_hash + sorted canonical JSON payload."""
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    content_to_hash = f"{prev_hash}:{canonical_json}".encode()
    return hashlib.sha256(content_to_hash).hexdigest()


class AuditLedger:
    """Append-only cryptographic hash-chained ledger storing immutable governance decisions."""

    def __init__(self, ledger_file: str | Path = "data/audit/audit_ledger.jsonl") -> None:
        self.ledger_path = Path(ledger_file)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.ledger_path.exists():
            self.ledger_path.touch()

    def get_latest_hash(self) -> tuple[str, int]:
        """Retrieve the hash and sequence number of the most recent ledger entry."""
        if not self.ledger_path.exists() or self.ledger_path.stat().st_size == 0:
            return GENESIS_PREV_HASH, -1

        last_line = ""
        with open(self.ledger_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()

        if not last_line:
            return GENESIS_PREV_HASH, -1

        entry_data = json.loads(last_line)
        return entry_data["entry_hash"], entry_data["sequence_num"]

    def append_decision(
        self,
        batch_id: str,
        payload: dict[str, Any],
        timestamp: float | None = None,
    ) -> AuditEntry:
        """Append an immutable decision record to the hash chain."""
        prev_hash, last_seq = self.get_latest_hash()
        next_seq = last_seq + 1
        ts = timestamp if timestamp is not None else time.time()

        entry_hash = compute_canonical_hash(prev_hash, payload)

        entry = AuditEntry(
            sequence_num=next_seq,
            timestamp=ts,
            batch_id=batch_id,
            prev_hash=prev_hash,
            payload=payload,
            entry_hash=entry_hash,
        )

        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

        return entry

    def read_all_entries(self) -> list[AuditEntry]:
        """Read all ledger entries sequentially."""
        if not self.ledger_path.exists() or self.ledger_path.stat().st_size == 0:
            return []

        entries = []
        with open(self.ledger_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(AuditEntry.model_validate_json(line.strip()))
        return entries


def verify_audit_ledger(ledger_path: str | Path) -> tuple[bool, list[str]]:
    """Cryptographically verify the integrity of the audit ledger from genesis.

    Returns
    -------
    (is_valid, list_of_violations)
    """
    path = Path(ledger_path)
    if not path.exists() or path.stat().st_size == 0:
        return True, []

    violations = []
    expected_prev_hash = GENESIS_PREV_HASH
    expected_seq = 0

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                entry = AuditEntry.model_validate_json(line.strip())
            except Exception as e:
                violations.append(f"Line {line_no}: Malformed JSON entry: {str(e)}")
                continue

            # 1. Check sequence number continuity
            if entry.sequence_num != expected_seq:
                violations.append(
                    f"Line {line_no} (Seq {entry.sequence_num}): Sequence break! Expected {expected_seq}."
                )

            # 2. Check parent hash linkage
            if entry.prev_hash != expected_prev_hash:
                violations.append(
                    f"Line {line_no} (Seq {entry.sequence_num}): Broken hash chain! "
                    f"Expected prev_hash {expected_prev_hash[:12]}..., got {entry.prev_hash[:12]}..."
                )

            # 3. Recompute canonical payload hash
            recomputed_hash = compute_canonical_hash(entry.prev_hash, entry.payload)
            if entry.entry_hash != recomputed_hash:
                violations.append(
                    f"Line {line_no} (Seq {entry.sequence_num}): Payload tampered! "
                    f"Recomputed hash {recomputed_hash[:12]}... != entry_hash {entry.entry_hash[:12]}..."
                )

            expected_prev_hash = entry.entry_hash
            expected_seq += 1

    return len(violations) == 0, violations
