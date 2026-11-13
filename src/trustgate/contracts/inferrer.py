"""Automated Data Contract inference for uncontracted datasets."""

import polars as pl

from trustgate.contracts.models import (
    ColumnContract,
    ColumnDType,
    DataContract,
    TrustLevel,
)
from trustgate.ingest.schema import infer_series_type


def infer_contract(
    df: pl.DataFrame,
    dataset_id: str = "inferred_dataset",
    name: str = "auto_inferred_contract",
) -> DataContract:
    """Infer a baseline DataContract from a sample or reference DataFrame.

    Marks the resulting contract with trust_level='INFERRED' (lower trust tier).

    Parameters
    ----------
    df : pl.DataFrame
        Source DataFrame to infer schema and constraints from.
    dataset_id : str
        Dataset identifier.
    name : str
        Contract name.

    Returns
    -------
    DataContract
        Draft contract specification.
    """
    column_contracts: list[ColumnContract] = []

    for col_name in df.columns:
        series = df[col_name]
        semantic_type = infer_series_type(series)

        # Map semantic type to ColumnDType
        if semantic_type == "numeric":
            dtype = ColumnDType.NUMERIC
            non_null = series.drop_nulls()
            min_val = (
                float(non_null.min()) if len(non_null) > 0 and non_null.dtype.is_numeric() else None
            )
            max_val = (
                float(non_null.max()) if len(non_null) > 0 and non_null.dtype.is_numeric() else None
            )
            allowed_vals = None
        elif semantic_type == "boolean":
            dtype = ColumnDType.BOOLEAN
            min_val = None
            max_val = None
            allowed_vals = None
        elif semantic_type == "datetime":
            dtype = ColumnDType.DATETIME
            min_val = None
            max_val = None
            allowed_vals = None
        elif semantic_type == "categorical":
            dtype = ColumnDType.CATEGORICAL
            min_val = None
            max_val = None
            # Extract distinct categories if <= 25
            distinct = series.drop_nulls().unique().head(25).to_list()
            allowed_vals = [str(x) for x in distinct]
        else:
            dtype = ColumnDType.TEXT
            min_val = None
            max_val = None
            allowed_vals = None

        col_contract = ColumnContract(
            name=col_name,
            dtype=dtype,
            required=True,
            nullable=series.null_count() > 0,
            min_value=min_val,
            max_value=max_val,
            allowed_values=allowed_vals,
        )
        column_contracts.append(col_contract)

    return DataContract(
        name=name,
        version="0.1.0-inferred",
        dataset_id=dataset_id,
        trust_level=TrustLevel.INFERRED,
        columns=column_contracts,
        description="Automatically inferred draft contract from batch data (Subject to human review)",
    )
