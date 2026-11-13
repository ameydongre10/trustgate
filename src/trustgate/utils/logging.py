"""Structured JSON logging with automatic PII and credential redaction."""

import json
import logging
import os
import re
import sys
from datetime import UTC, datetime
from typing import Any

# Common patterns for credentials, tokens, and PII to redact in logs
REDACTION_PATTERNS = [
    (
        re.compile(
            r'(?i)(api[-_]?key|secret|token|password|auth|bearer)\s*[:=]\s*["\']?([^"\'\s]+)["\']?'
        ),
        r'\1="[REDACTED]"',
    ),
    (re.compile(r"\b[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b"), "[REDACTED_AADHAAR]"),
    (re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b"), "[REDACTED_PAN]"),
    (
        re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"),
        "[REDACTED_CC]",
    ),
]


def redact_sensitive_text(text: str) -> str:
    """Redact sensitive credentials and direct identifiers from text."""
    redacted = text
    for pattern, replacement in REDACTION_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


class JSONFormatter(logging.Formatter):
    """Formats log records as structured, single-line JSON with redaction."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_sensitive_text(record.getMessage()),
            "module": record.module,
            "lineno": record.lineno,
        }

        # Include custom extra fields if provided
        if hasattr(record, "batch_id"):
            log_entry["batch_id"] = record.batch_id
        if hasattr(record, "action"):
            log_entry["action"] = record.action
        if hasattr(record, "details"):
            log_entry["details"] = record.details

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def get_logger(name: str = "trustgate") -> logging.Logger:
    """Get a configured structured JSON logger.

    Parameters
    ----------
    name : str
        The module/component name.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        level_name = os.environ.get("TRUSTGATE_LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level_name, logging.INFO))

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False

    return logger
