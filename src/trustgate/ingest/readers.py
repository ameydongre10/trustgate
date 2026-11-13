"""Safe streaming readers for CSV, TSV, Parquet, and JSONL data files."""

import io
import os
import zipfile
from collections.abc import Iterator
from pathlib import Path

import polars as pl
from pydantic import BaseModel, Field

from trustgate.ingest.sanitizer import sanitize_column_names, sanitize_path
from trustgate.utils.errors import DecompressionBombError, IngestionError

# Maximum allowable decompression ratio (e.g. 100:1) to prevent zip/gzip bombs
MAX_DECOMPRESSION_RATIO = 100.0
# Maximum file size allowed for in-memory read (500 MB by default)
DEFAULT_MAX_FILE_BYTES = 500 * 1024 * 1024
DEFAULT_CHUNK_SIZE = 100_000


class BatchMetadata(BaseModel):
    file_path: str | None = None
    file_format: str
    file_size_bytes: int
    row_count: int
    column_count: int
    column_names: list[str]
    warnings: list[str] = Field(default_factory=list)
    encoding: str = "utf-8"


class BatchData:
    """Encapsulates parsed batch data, metadata, and column sanitization records."""

    def __init__(
        self,
        df: pl.DataFrame,
        metadata: BatchMetadata,
        raw_columns: list[str],
    ) -> None:
        self.df = df
        self.metadata = metadata
        self.raw_columns = raw_columns

    @property
    def row_count(self) -> int:
        return self.metadata.row_count

    @property
    def column_count(self) -> int:
        return self.metadata.column_count


def check_decompression_safety(file_path: Path) -> None:
    """Check archive or compressed file against decompression bomb ratios."""
    compressed_size = file_path.stat().st_size
    if compressed_size == 0:
        return

    # Check zip archives
    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path, "r") as zf:
            uncompressed_size = sum(info.file_size for info in zf.infolist())
            ratio = uncompressed_size / compressed_size
            if ratio > MAX_DECOMPRESSION_RATIO:
                raise DecompressionBombError(
                    f"Zip compression ratio {ratio:.1f}:1 exceeds maximum safe threshold {MAX_DECOMPRESSION_RATIO}:1",
                    details={
                        "compressed_size": compressed_size,
                        "uncompressed_size": uncompressed_size,
                        "ratio": ratio,
                    },
                )

    # Check gzip files (.gz)
    elif file_path.suffix.lower() == ".gz":
        with open(file_path, "rb") as f:
            f.seek(-4, os.SEEK_END)
            # Last 4 bytes of gzip contain ISIZE (uncompressed size modulo 2^32)
            uncompressed_isize = int.from_bytes(f.read(4), byteorder="little")
            ratio = uncompressed_isize / compressed_size if compressed_size > 0 else 0
            if ratio > MAX_DECOMPRESSION_RATIO:
                raise DecompressionBombError(
                    f"Gzip compression ratio {ratio:.1f}:1 exceeds maximum safe threshold {MAX_DECOMPRESSION_RATIO}:1",
                    details={
                        "compressed_size": compressed_size,
                        "uncompressed_size": uncompressed_isize,
                        "ratio": ratio,
                    },
                )


def detect_encoding(file_path: Path) -> str:
    """Detect text encoding with BOM and Latin-1 fallback."""
    with open(file_path, "rb") as f:
        head = f.read(4)

    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    elif head.startswith(b"\xff\xfe"):
        return "utf-16-le"
    elif head.startswith(b"\xfe\xff"):
        return "utf-16-be"

    # Read first 16KB to check UTF-8 vs Latin-1 validity
    with open(file_path, "rb") as f:
        sample = f.read(16384)

    try:
        sample.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def read_batch_file(
    file_path: Path | str,
    base_dir: Path | str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> BatchData:
    """Safely ingest tabular data from CSV, TSV, Parquet, or JSONL into a BatchData object.

    Parameters
    ----------
    file_path : Path | str
        Path to the data file.
    base_dir : Optional[Path | str]
        Boundary folder for path traversal prevention.
    chunk_size : int
        Chunk size for batch processing.
    max_file_bytes : int
        Maximum permitted file size in bytes.

    Returns
    -------
    BatchData
        Parsed DataFrame and metadata.
    """
    path = sanitize_path(file_path, base_dir)
    if not path.exists():
        raise IngestionError(f"File not found: {path}")

    file_size = path.stat().st_size
    if file_size > max_file_bytes:
        raise IngestionError(
            f"File size {file_size} bytes exceeds maximum permitted size {max_file_bytes} bytes",
            details={"file_size": file_size, "max_file_bytes": max_file_bytes},
        )

    # Decompression bomb safety check
    check_decompression_safety(path)

    ext = path.suffix.lower()
    warnings: list[str] = []

    # Handle completely empty file (0 bytes)
    if file_size == 0:
        empty_df = pl.DataFrame()
        metadata = BatchMetadata(
            file_path=str(path),
            file_format=ext.lstrip("."),
            file_size_bytes=0,
            row_count=0,
            column_count=0,
            column_names=[],
            warnings=["File is 0 bytes (empty file)"],
        )
        return BatchData(empty_df, metadata, raw_columns=[])

    try:
        if ext in (".parquet", ".pq"):
            df = pl.read_parquet(path)
            raw_cols = df.columns
            clean_cols, col_warnings = sanitize_column_names(raw_cols)
            warnings.extend(col_warnings)
            if clean_cols != raw_cols:
                df.columns = clean_cols
            encoding = "binary"

        elif ext in (".csv", ".tsv", ".txt"):
            separator = "\t" if ext == ".tsv" else ","
            encoding = detect_encoding(path)

            try:
                if encoding not in ("utf-8", "utf-8-sig"):
                    # For non-UTF8 encodings (latin-1, etc.), read text and feed buffer to Polars
                    text_content = path.read_text(encoding=encoding)
                    df = pl.read_csv(
                        io.StringIO(text_content),
                        separator=separator,
                        infer_schema_length=10000,
                        ignore_errors=True,
                        truncate_ragged_lines=True,
                        null_values=["", "NA", "N/A", "null", "NULL", "None", "-999", "?"],
                    )
                else:
                    df = pl.read_csv(
                        path,
                        separator=separator,
                        infer_schema_length=10000,
                        ignore_errors=True,
                        truncate_ragged_lines=True,
                        null_values=["", "NA", "N/A", "null", "NULL", "None", "-999", "?"],
                    )
            except pl.exceptions.NoDataError:
                # Header-only or 0 rows
                df = pl.DataFrame()

            raw_cols = df.columns
            clean_cols, col_warnings = sanitize_column_names(raw_cols)
            warnings.extend(col_warnings)
            if clean_cols != raw_cols and len(clean_cols) == len(df.columns):
                df.columns = clean_cols

        elif ext in (".jsonl", ".ndjson"):
            encoding = detect_encoding(path)
            df = pl.read_ndjson(path)
            raw_cols = df.columns
            clean_cols, col_warnings = sanitize_column_names(raw_cols)
            warnings.extend(col_warnings)
            if clean_cols != raw_cols:
                df.columns = clean_cols

        else:
            raise IngestionError(f"Unsupported file format extension: {ext}")

    except Exception as e:
        if isinstance(e, (IngestionError, DecompressionBombError)):
            raise
        raise IngestionError(
            f"Failed to parse file {path.name}: {str(e)}", details={"error": str(e)}
        ) from e

    metadata = BatchMetadata(
        file_path=str(path),
        file_format=ext.lstrip("."),
        file_size_bytes=file_size,
        row_count=len(df),
        column_count=len(df.columns),
        column_names=df.columns,
        warnings=warnings,
        encoding=encoding,
    )

    return BatchData(df=df, metadata=metadata, raw_columns=raw_cols)


def stream_batch_chunks(
    file_path: Path | str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[pl.DataFrame]:
    """Stream large CSV or Parquet files in memory-bounded DataFrame chunks."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in (".parquet", ".pq"):
        # Scan parquet lazily and slice
        lazy_df = pl.scan_parquet(path)
        total_rows = lazy_df.select(pl.len()).collect().item()
        for offset in range(0, total_rows, chunk_size):
            yield lazy_df.slice(offset, chunk_size).collect()

    elif ext in (".csv", ".tsv"):
        separator = "\t" if ext == ".tsv" else ","
        lazy_df = pl.scan_csv(
            path,
            separator=separator,
            ignore_errors=True,
            null_values=["", "NA", "N/A", "null", "NULL", "None"],
        )
        total_rows = lazy_df.select(pl.len()).collect().item()
        for offset in range(0, total_rows, chunk_size):
            yield lazy_df.slice(offset, chunk_size).collect()
    else:
        # Fallback to single chunk
        batch = read_batch_file(path)
        yield batch.df
