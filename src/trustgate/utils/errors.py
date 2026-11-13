"""Structured exception hierarchy for TrustGate."""

from enum import StrEnum
from typing import Any


class FailSafeAction(StrEnum):
    """Fail-safe default actions when internal errors or timeouts occur."""

    QUARANTINE = "QUARANTINE"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"


class TrustGateError(Exception):
    """Base exception for all TrustGate runtime errors.

    Attributes
    ----------
    message : str
        Human-readable error description.
    code : str
        Standardized error code.
    fail_safe_action : FailSafeAction
        Action to execute if this error occurs during gating.
    details : Dict[str, Any]
        Additional context or metadata.
    """

    def __init__(
        self,
        message: str,
        code: str = "ERR_INTERNAL",
        fail_safe_action: FailSafeAction = FailSafeAction.QUARANTINE,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.fail_safe_action = fail_safe_action
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "fail_safe_action": self.fail_safe_action.value,
            "details": self.details,
        }


class IngestionError(TrustGateError):
    """Raised when data cannot be safely read or parsed."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="ERR_INGEST_FAILURE",
            fail_safe_action=FailSafeAction.QUARANTINE,
            details=details,
        )


class DecompressionBombError(IngestionError):
    """Raised when an uncompressed file exceeds compression safety ratio limits."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=f"Decompression bomb detected: {message}",
            details=details,
        )
        self.code = "ERR_DECOMPRESSION_BOMB"


class ContractViolationError(TrustGateError):
    """Raised when data explicitly breaches a non-repairable contract specification."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="ERR_CONTRACT_VIOLATION",
            fail_safe_action=FailSafeAction.QUARANTINE,
            details=details,
        )


class ProfilingError(TrustGateError):
    """Raised when statistical profiling fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="ERR_PROFILING_FAILURE",
            fail_safe_action=FailSafeAction.ESCALATE_TO_HUMAN,
            details=details,
        )


class PIIValidationError(TrustGateError):
    """Raised when prohibited direct PII is detected."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="ERR_PII_VIOLATION",
            fail_safe_action=FailSafeAction.QUARANTINE,
            details=details,
        )


class PredictionError(TrustGateError):
    """Raised when damage prediction encounters numerical or model failures."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="ERR_DAMAGE_PREDICTION",
            fail_safe_action=FailSafeAction.ESCALATE_TO_HUMAN,
            details=details,
        )


class OutOfDistributionError(PredictionError):
    """Raised when issue profile features are outside the damage predictor support."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=f"Issue profile is Out-Of-Distribution (OOD): {message}",
            details=details,
        )
        self.code = "ERR_PREDICTOR_OOD"
        self.fail_safe_action = FailSafeAction.ESCALATE_TO_HUMAN


class RepairError(TrustGateError):
    """Raised when automated repair fails or violates budget/integrity."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="ERR_REPAIR_FAILURE",
            fail_safe_action=FailSafeAction.QUARANTINE,
            details=details,
        )


class AuditTamperingError(TrustGateError):
    """Raised when cryptographic verification detects audit ledger modification."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=f"CRITICAL: Audit log integrity breached! {message}",
            code="ERR_AUDIT_TAMPERED",
            fail_safe_action=FailSafeAction.QUARANTINE,
            details=details,
        )


class SecurityError(TrustGateError):
    """Raised on security violations (formula injection, path traversal, unauthorized review)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=message,
            code="ERR_SECURITY_VIOLATION",
            fail_safe_action=FailSafeAction.QUARANTINE,
            details=details,
        )
