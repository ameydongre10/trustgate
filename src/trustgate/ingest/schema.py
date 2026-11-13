"""Safe type inference, numeric normalization, and boolean/datetime coercion."""

import re
from datetime import UTC, datetime
from typing import Any

import polars as pl

# Common null sentinel strings
NULL_SENTINELS = {"", "na", "n/a", "null", "none", "nan", "\\n", "nil", "?", "-", "#n/a", "#na"}

BOOLEAN_TRUE_VALUES = {"true", "t", "yes", "y", "1"}
BOOLEAN_FALSE_VALUES = {"false", "f", "no", "n", "0"}

NUMERIC_CLEAN_RE = re.compile(r"[$€£¥,%\s]")


def parse_boolean_value(val: Any) -> bool | None:
    """Coerce boolean variants (Y/N, yes/no, 0/1, true/false) safely."""
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in NULL_SENTINELS:
        return None
    if s in BOOLEAN_TRUE_VALUES:
        return True
    if s in BOOLEAN_FALSE_VALUES:
        return False
    return None


def parse_numeric_string(val: Any) -> float | None:
    """Parse numeric values with currency symbols, percentages, and locale delimiters.

    Handles:
    - Standard US: '1,234.56' -> 1234.56
    - German/EU: '1.234,56' -> 1234.56
    - Percentages: '45.2%' -> 45.2 (or 0.452)
    - Currency: '$1,200.00' -> 1200.0
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip()
    if not s or s.lower() in NULL_SENTINELS:
        return None

    # Check for European format with '.' as thousands and ',' as decimal (e.g. 1.234,56)
    if re.match(r"^-?[\$€£¥]?\s*\d{1,3}(?:\.\d{3})*,\d+$", s):
        clean = re.sub(r"[\$€£¥\s]", "", s).replace(".", "").replace(",", ".")
        try:
            return float(clean)
        except ValueError:
            return None

    # Remove standard currency, commas, spaces
    clean = NUMERIC_CLEAN_RE.sub("", s)
    try:
        return float(clean)
    except ValueError:
        return None


def parse_timestamp_value(val: Any) -> datetime | None:
    """Parse diverse date and timestamp formats, epoch seconds, and epoch milliseconds."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=UTC)

    # Check for integer/float epoch timestamps
    if isinstance(val, (int, float)) or (isinstance(val, str) and val.strip().isdigit()):
        num = float(val)
        # If timestamp is likely milliseconds (> 10^11, e.g. 1670000000000 is year 2022)
        if num > 1e11:
            return datetime.fromtimestamp(num / 1000.0, tz=UTC)
        elif num > 0:
            return datetime.fromtimestamp(num, tz=UTC)

    s = str(val).strip()
    if not s or s.lower() in NULL_SENTINELS:
        return None

    # Try common ISO and datetime formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s.split(".")[0], fmt)
            return dt.replace(tzinfo=UTC)
        except ValueError:
            continue

    return None


def infer_series_type(series: pl.Series) -> str:
    """Infer semantic column type (numeric, boolean, datetime, categorical, text)."""
    dtype = series.dtype

    if dtype.is_numeric():
        return "numeric"
    elif dtype.is_temporal():
        return "datetime"
    elif dtype == pl.Boolean:
        return "boolean"

    # Sample non-null strings to infer semantic type
    non_nulls = series.drop_nulls()
    if len(non_nulls) == 0:
        return "null"

    sample_size = min(len(non_nulls), 100)
    samples = non_nulls.head(sample_size).to_list()

    # Check boolean representation
    bool_matches = sum(1 for x in samples if parse_boolean_value(x) is not None)
    if bool_matches / sample_size > 0.9:
        return "boolean"

    # Check numeric representation
    num_matches = sum(1 for x in samples if parse_numeric_string(x) is not None)
    if num_matches / sample_size > 0.9:
        return "numeric"

    # Check datetime representation
    dt_matches = sum(1 for x in samples if parse_timestamp_value(x) is not None)
    if dt_matches / sample_size > 0.8:
        return "datetime"

    # Distinguish categorical (low unique ratio) vs free text
    n_unique = series.n_unique()
    if n_unique <= 50 or (len(series) > 0 and (n_unique / len(series)) < 0.05):
        return "categorical"

    return "text"
