import logging
import uuid
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import ValidationError

from app.module.process_claim.agents.improved_adk_agents import run_claim_processing_pipeline
from app.module.process_claim.schemas.schemas import (
    BillDocument,
    ClaimDecision,
    DischargeSummaryDocument,
    ProcessClaimResponse,
    ValidationResult,
)
from app.module.process_claim.services.ocr import process_ocr

process_claim_router = APIRouter()


@process_claim_router.post("/process-claim", response_model=ProcessClaimResponse)
async def process_claim_documents(files: List[UploadFile] = File(...)):
    ocr_texts = []

    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {file.filename}. Only PDF files are allowed.",
            )
        if file.filename is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is missing a filename.",
            )

        content = await file.read()
        ocr_text = await process_ocr(content, file.filename)
        ocr_texts.append(ocr_text)

    try:
        agent_results = await run_claim_processing_pipeline(ocr_texts, user_id=str(uuid.uuid4()))
    except Exception as e:
        logging.error(f"Agent pipeline error: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error") from e

    # Parse and validate agent output into Pydantic models
    documents = []
    all_missing_documents = []
    all_discrepancies = []
    claim_decisions = []

    logging.info(f"Processing {len(agent_results)} agent results")
    for i, result in enumerate(agent_results):
        logging.info(f"Agent result {i}: {result}")

        # Handle case where result might be a string or have unexpected structure
        if isinstance(result, str):
            logging.warning(f"Agent returned string instead of dict: {result}")
            continue

        if not isinstance(result, dict):
            logging.warning(f"Agent returned unexpected type: {type(result)}")
            continue

        # Extract documents - assuming 'extracted_fields' contains document info
        extracted = result.get("extracted_fields", {})
        doc_type = extracted.get("type")
        logging.info(f"Extracted fields: {extracted}, doc_type: {doc_type}")

        try:
            if doc_type == "bill":
                documents.append(BillDocument(**extracted))
            elif doc_type == "discharge_summary":
                documents.append(DischargeSummaryDocument(**extracted))
            else:
                logging.warning(f"Unknown document type: {doc_type}")
        except ValidationError as ve:
            logging.error(f"Validation error for document: {ve}")

        # Aggregate validation results
        validation_result = result.get("validation_result", {})
        if isinstance(validation_result, dict):
            missing_docs = validation_result.get("missing_documents", [])
            discrepancies = validation_result.get("discrepancies", [])

            if isinstance(missing_docs, list):
                all_missing_documents.extend(missing_docs)
            if isinstance(discrepancies, list):
                all_discrepancies.extend(discrepancies)

        # Collect claim decisions
        claim_decision = result.get("claim_decision", {})
        if isinstance(claim_decision, dict):
            claim_decisions.append(claim_decision)

    # Create aggregated validation result
    validation = ValidationResult(
        missing_documents=list(set(all_missing_documents)),  # Remove duplicates
        discrepancies=list(set(all_discrepancies)),  # Remove duplicates
    )

    # Determine final claim decision based on all results
    if not claim_decisions:
        claim_decision = ClaimDecision(status="rejected", reason="No claim decision returned by agent")
    else:
        # If any decision is rejected, overall decision is rejected
        any_rejected = any(decision.get("status") == "rejected" for decision in claim_decisions)
        if any_rejected:
            # Find the first rejection reason
            rejection_reason = next(
                (decision.get("reason", "Unknown rejection reason") for decision in claim_decisions if decision.get("status") == "rejected"),
                "Multiple documents have validation issues",
            )
            claim_decision = ClaimDecision(status="rejected", reason=rejection_reason)
        else:
            # All decisions are approved
            claim_decision = ClaimDecision(status="approved", reason="All required documents present and data is consistent")

    return ProcessClaimResponse(
        documents=documents,
        validation=validation,
        claim_decision=claim_decision,
    )
