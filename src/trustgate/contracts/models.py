"""Data Contract specification models and schemas."""

from enum import StrEnum

from pydantic import BaseModel, Field


class TrustLevel(StrEnum):
    VERIFIED = "VERIFIED"
    INFERRED = "INFERRED"


class ColumnDType(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"


class PIIType(StrEnum):
    NONE = "none"
    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    AADHAAR = "indian_aadhaar"
    PAN = "indian_pan"
    UPI = "upi_id"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"


class ColumnContract(BaseModel):
    name: str
    dtype: ColumnDType
    required: bool = True
    nullable: bool = True
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[str] | None = None
    regex_pattern: str | None = None
    pii_type: PIIType = PIIType.NONE
    description: str | None = None


class BreakingChangePolicy(BaseModel):
    allow_added_optional_columns: bool = True
    allow_reordered_columns: bool = True
    allow_widened_types: bool = True
    allow_removed_optional_columns: bool = True
    allow_removed_required_columns: bool = False
    allow_narrowed_types: bool = False


class DataContract(BaseModel):
    name: str
    version: str = "1.0.0"
    dataset_id: str
    owner: str = "data-governance@trustgate.ai"
    freshness_hours: int | None = None
    description: str | None = None
    trust_level: TrustLevel = TrustLevel.VERIFIED
    breaking_policy: BreakingChangePolicy = Field(default_factory=BreakingChangePolicy)
    columns: list[ColumnContract] = Field(default_factory=list)

    def get_column(self, name: str) -> ColumnContract | None:
        for col in self.columns:
            if col.name.lower() == name.lower():
                return col
        return None
