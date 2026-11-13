"""Property-based tests for cryptographic audit ledger integrity and tamper detection."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trustgate.audit.ledger import AuditLedger, verify_audit_ledger


@settings(max_examples=25, deadline=None)
@given(
    payloads=st.lists(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz"),
            values=st.one_of(st.integers(min_value=0, max_value=1000), st.text(max_size=20)),
            min_size=1,
            max_size=5,
        ),
        min_size=1,
        max_size=15,
    )
)
def test_property_audit_chain_always_valid_on_honest_writes(
    tmp_path_factory: pytest.TempPathFactory,
    payloads: list[dict],
) -> None:
    """Property: Any sequence of valid appends preserves complete cryptographic chain validity."""
    temp_dir = tmp_path_factory.mktemp("ledger_prop")
    ledger_file = temp_dir / "prop_ledger.jsonl"
    ledger = AuditLedger(ledger_file=ledger_file)

    for idx, p in enumerate(payloads):
        ledger.append_decision(batch_id=f"b_{idx}", payload=p)

    is_valid, violations = verify_audit_ledger(ledger_file)
    assert is_valid
    assert len(violations) == 0
