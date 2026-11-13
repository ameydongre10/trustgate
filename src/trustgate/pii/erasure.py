"""Right-to-Erasure (RTBF) hooks and subject record tombstoning."""

import hashlib
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ErasureRequest(BaseModel):
    subject_identifier: str  # e.g. Email or Aadhaar or User ID
    dataset_id: str
    reason: str
    requester_id: str = "dpo@trustgate.ai"
    requested_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class TombstoneReceipt(BaseModel):
    tombstone_id: str
    subject_hash: str
    dataset_id: str
    rows_tombstoned_count: int
    executed_at: str
    status: str = "TOMBSTONED"
    audit_event: dict[str, Any]


def execute_right_to_erasure(
    request: ErasureRequest,
    target_records: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], TombstoneReceipt]:
    """Execute GDPR / DPDP right-to-erasure by tombstoning subject records.

    Parameters
    ----------
    request : ErasureRequest
        Erasure request parameters.
    target_records : Optional[list[dict[str, Any]]]
        List of records in quarantine or cache to sanitize.

    Returns
    -------
    tuple[list[dict[str, Any]], TombstoneReceipt]
        (sanitized_records, tombstone_receipt)
    """
    subject_val = request.subject_identifier.strip().lower()
    subject_hash = hashlib.sha256(subject_val.encode("utf-8")).hexdigest()

    records = target_records or []
    sanitized: list[dict[str, Any]] = []
    affected_count = 0

    for rec in records:
        # Check if record contains the subject identifier
        rec_str = str(rec).lower()
        if subject_val in rec_str:
            affected_count += 1
            # Replace record payload with an immutable tombstone marker
            tombstone_rec = {
                "_tombstone": True,
                "_tombstoned_at": datetime.now(UTC).isoformat(),
                "_subject_hash": subject_hash,
                "_reason": request.reason,
            }
            sanitized.append(tombstone_rec)
        else:
            sanitized.append(rec)

    executed_at = datetime.now(UTC).isoformat()
    tombstone_id = (
        f"tomb_{hashlib.sha256(f'{subject_hash}_{executed_at}'.encode()).hexdigest()[:16]}"
    )

    audit_event = {
        "event_type": "RIGHT_TO_ERASURE_EXECUTED",
        "tombstone_id": tombstone_id,
        "subject_hash": subject_hash,
        "dataset_id": request.dataset_id,
        "requester_id": request.requester_id,
        "rows_affected": affected_count,
        "timestamp_utc": executed_at,
    }

    receipt = TombstoneReceipt(
        tombstone_id=tombstone_id,
        subject_hash=subject_hash,
        dataset_id=request.dataset_id,
        rows_tombstoned_count=affected_count,
        executed_at=executed_at,
        audit_event=audit_event,
    )

    return sanitized, receipt
