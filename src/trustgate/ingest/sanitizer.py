"""Sanitization utilities for column headers, cell values, and file paths."""

import re
from pathlib import Path

# Formula injection prefixes that spreadsheet programs execute
FORMULA_INJECTION_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

# Dangerous command execution patterns in strings
SUSPICIOUS_CMD_PATTERN = re.compile(r"(?i)(cmd\||powershell\||regsvr32|mshta|wscript)")

# Prompt injection patterns in metadata/column headers
PROMPT_INJECTION_PATTERN = re.compile(
    r"(?i)(ignore (all )?previous instructions|system prompt|you are now|disregard|admin override|bypass policy)"
)


def sanitize_path(file_path: str | Path, base_dir: Path | str | None = None) -> Path:
    """Validate and sanitize file paths to prevent path traversal and symlink attacks.

    Parameters
    ----------
    file_path : str | Path
        Path to validate.
    base_dir : Path | str | None
        Optional directory boundary that the path must reside within.

    Returns
    -------
    Path
        Resolved, validated Path.

    Raises
    ------
    ValueError
        If path attempts traversal outside base_dir or points to an unsafe symlink.
    """
    path_obj = Path(file_path)
    # Check for relative traversal components in raw string representation
    raw_str = str(file_path)
    if ".." in raw_str:
        resolved = path_obj.resolve()
        if base_dir:
            base_resolved = Path(base_dir).resolve()
            if not str(resolved).startswith(str(base_resolved)):
                raise ValueError(f"Path traversal detected: {file_path} outside {base_dir}")

    # Check symlinks if the file exists
    if path_obj.is_symlink():
        target = path_obj.resolve()
        if base_dir:
            base_resolved = Path(base_dir).resolve()
            if not str(target).startswith(str(base_resolved)):
                raise ValueError(f"Unsafe symlink detected: {file_path} points outside boundary")

    return path_obj


def sanitize_column_names(columns: list[str]) -> tuple[list[str], list[str]]:
    """Sanitize column headers for security, duplicates, empty values, and formula injection.

    Parameters
    ----------
    columns : list[str]
        Raw column names.

    Returns
    -------
    tuple[list[str], list[str]]
        (sanitized_columns, warnings)
    """
    sanitized: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for idx, col in enumerate(columns):
        raw_name = str(col)
        name = raw_name.strip()

        # Handle empty names
        if not name:
            name = f"unnamed_column_{idx}"
            warnings.append(f"Empty column name at index {idx} replaced with '{name}'")

        # Check for prompt injection in column names
        if PROMPT_INJECTION_PATTERN.search(name):
            warnings.append(f"Prompt injection pattern detected in header: '{name}'. Sanitized.")
            name = re.sub(PROMPT_INJECTION_PATTERN, "[FILTERED_PROMPT]", name)

        # Check and neutralize formula injection prefixes
        if name.startswith(FORMULA_INJECTION_PREFIXES):
            warnings.append(f"Formula injection prefix stripped from header: '{name}'")
            name = name.lstrip("=+-@\t\r ")
            if not name:
                name = f"sanitized_col_{idx}"

        # Check for suspicious OS command patterns
        if SUSPICIOUS_CMD_PATTERN.search(name):
            warnings.append(f"Dangerous command pattern stripped from header: '{name}'")
            name = re.sub(SUSPICIOUS_CMD_PATTERN, "", name)

        # Truncate overly long names (DOS defense against 10k character header strings)
        if len(name) > 128:
            warnings.append(f"Column name '{name[:30]}...' exceeded 128 chars; truncated.")
            name = name[:128]

        # Handle duplicates with deterministic suffix
        base_name = name
        counter = 1
        while name.lower() in seen:
            name = f"{base_name}_{counter}"
            counter += 1
            if counter == 2:
                warnings.append(f"Duplicate column '{base_name}' disambiguated to '{name}'")

        seen.add(name.lower())
        sanitized.append(name)

    return sanitized, warnings
