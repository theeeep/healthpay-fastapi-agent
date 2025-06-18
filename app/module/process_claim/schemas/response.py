from datetime import date
from typing import List, Literal, Union

from pydantic import BaseModel, Field


class BillDocument(BaseModel):
    type: Literal["bill"]
    hospital_name: str
    total_amount: float
    date_of_service: date


class DischargeSummaryDocument(BaseModel):
    type: Literal["discharge_summary"]
    patient_name: str
    diagnosis: str
    admission_date: date
    discharge_date: date


# Union type to allow for different document types
Document = Union[BillDocument, DischargeSummaryDocument]


class ValidationResult(BaseModel):
    missing_documents: List[str] = Field(default_factory=list)
    discrepancies: List[str] = Field(default_factory=list)


class ClaimDecision(BaseModel):
    status: Literal["approved", "rejected"]
    reason: str


class ProcessClaimResponse(BaseModel):
    documents: List[Document]
    validation: ValidationResult
    claim_decision: ClaimDecision
