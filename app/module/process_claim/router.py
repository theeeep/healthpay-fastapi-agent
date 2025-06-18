import logging
import uuid
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import ValidationError

from app.core.logger import logger
from app.module.process_claim.agents.genai_agents import run_claim_processing_pipeline as run_genai_pipeline
from app.module.process_claim.agents.improved_adk_agents import run_claim_processing_pipeline as run_adk_pipeline
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
        # Step 1: Use LLM (Gemini) for classification, extraction, and basic validation
        logger.info("Step 1: Using LLM (Gemini) for classification, extraction, and basic validation")
        genai_results = await run_genai_pipeline(ocr_texts, user_id=str(uuid.uuid4()))
        logger.info(f"GenAI pipeline returned {len(genai_results)} results")

        # Step 2: Use ADK agents for enhanced validation and decision making (multi-agent orchestration)
        logger.info("Step 2: Using ADK agents for enhanced validation and decision making")
        adk_results = await run_adk_pipeline(ocr_texts, user_id=str(uuid.uuid4()))
        logger.info(f"ADK pipeline returned {len(adk_results)} results")

        # Combine results: Use GenAI for extraction, ADK for enhanced validation/decision
        agent_results = []

        # Add GenAI results (good extraction)
        for genai_result in genai_results:
            agent_results.append(genai_result)

        # Add ADK results (enhanced validation/decision) - but only if they provide value
        for adk_result in adk_results:
            # Only add ADK results if they have enhanced validation or decision making
            validation = adk_result.get("validation_result", {})
            decision = adk_result.get("claim_decision", {})

            if (
                validation.get("data_quality_score", 0) > 0
                or decision.get("confidence_score", 0) > 0
                or validation.get("recommendations")
                or decision.get("required_actions")
            ):
                agent_results.append(adk_result)

        logger.info(f"Combined {len(agent_results)} total results from both pipelines")

    except Exception as e:
        logger.error(f"Agent pipeline error: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error") from e

    # Parse and validate agent output into Pydantic models
    documents = []
    all_missing_documents = []
    all_discrepancies = []
    claim_decisions = []

    logger.info(f"Processing {len(agent_results)} agent results")
    for i, result in enumerate(agent_results):
        logger.info(f"Processing agent result {i + 1}/{len(agent_results)}")
        logger.info(f"Agent result {i} type: {type(result)}")
        logger.info(f"Agent result {i} keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")

        # Handle case where result might be a string or have unexpected structure
        if isinstance(result, str):
            logger.warning(f"Agent returned string instead of dict: {result}")
            continue

        if not isinstance(result, dict):
            logger.warning(f"Agent returned unexpected type: {type(result)}")
            continue

        # Extract documents - assuming 'extracted_fields' contains document info
        extracted = result.get("extracted_fields", {})
        logger.info(f"Extracted fields type: {type(extracted)}")
        logger.info(f"Extracted fields content: {extracted}")

        # Skip if no extracted fields (ADK pipeline only provides validation/decision)
        if extracted is None:
            logger.info(f"Skipping document processing for result {i} - no extracted fields (ADK validation/decision only)")
            # Still collect validation and decision data
            validation_result = result.get("validation_result", {})
            if isinstance(validation_result, dict):
                missing_docs = validation_result.get("missing_documents", [])
                discrepancies = validation_result.get("discrepancies", [])

                if isinstance(missing_docs, list):
                    all_missing_documents.extend(missing_docs)
                if isinstance(discrepancies, list):
                    all_discrepancies.extend(discrepancies)

            claim_decision = result.get("claim_decision", {})
            if isinstance(claim_decision, dict):
                claim_decisions.append(claim_decision)
            continue

        # Handle case where extracted might be a string
        if isinstance(extracted, str):
            logger.warning(f"Extracted fields is a string: {extracted}")
            continue

        if not isinstance(extracted, dict):
            logger.warning(f"Extracted fields is not a dict: {type(extracted)}")
            continue

        doc_type = extracted.get("type")
        logger.info(f"Document type: {doc_type}")

        # Quality check: Skip poor quality documents
        hospital_name = extracted.get("hospital_name", "")
        total_amount = extracted.get("total_amount", 0.0)
        patient_name = extracted.get("patient_name", "")

        # Skip documents with poor quality data
        if doc_type == "bill" and (hospital_name == "Unknown Hospital" or total_amount == 0.0):
            logger.info(f"Skipping poor quality bill document: hospital={hospital_name}, amount={total_amount}")
            continue

        if doc_type == "discharge_summary" and (patient_name == "Unknown Patient" or patient_name == ""):
            logger.info(f"Skipping poor quality discharge summary: patient={patient_name}")
            continue

        try:
            if doc_type == "bill":
                bill_doc = BillDocument(**extracted)
                documents.append(bill_doc)
                logger.info(f"Successfully created BillDocument: {bill_doc}")
            elif doc_type == "discharge_summary":
                discharge_doc = DischargeSummaryDocument(**extracted)
                documents.append(discharge_doc)
                logger.info(f"Successfully created DischargeSummaryDocument: {discharge_doc}")
            else:
                logger.warning(f"Unknown document type: {doc_type}")
        except ValidationError as ve:
            logger.error(f"Validation error for document: {ve}")
            logger.error(f"Failed document data: {extracted}")

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
    # Filter out discrepancies that are no longer relevant after quality filtering
    filtered_discrepancies = []
    for discrepancy in all_discrepancies:
        # Skip discrepancies about unknown hospitals or zero amounts if we have good quality docs
        if any(
            doc.hospital_name != "Unknown Hospital" and doc.total_amount > 0
            for doc in documents
            if hasattr(doc, "hospital_name") and hasattr(doc, "total_amount")
        ):
            if "Hospital name is unknown" in discrepancy or "Total amount is zero" in discrepancy:
                continue
        filtered_discrepancies.append(discrepancy)

    # Filter missing documents based on what we actually have
    filtered_missing_documents = []
    document_types_found = [doc.type for doc in documents]

    if "bill" not in document_types_found:
        filtered_missing_documents.append("bill")
    if "discharge_summary" not in document_types_found:
        filtered_missing_documents.append("discharge_summary")

    validation = ValidationResult(
        missing_documents=filtered_missing_documents,
        discrepancies=filtered_discrepancies,
    )

    # Determine final claim decision based on all results
    if not claim_decisions:
        claim_decision = ClaimDecision(status="rejected", reason="No claim decision returned by agent")
    else:
        # Prioritize approved decisions if we have good quality documents
        has_good_quality_documents = any(
            doc.hospital_name != "Unknown Hospital" and doc.total_amount > 0
            for doc in documents
            if hasattr(doc, "hospital_name") and hasattr(doc, "total_amount")
        )

        # If we have good quality documents, prioritize approval
        if has_good_quality_documents:
            # Check if any decision is rejected
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
        else:
            # No good quality documents, use the most conservative decision
            any_rejected = any(decision.get("status") == "rejected" for decision in claim_decisions)
            if any_rejected:
                # Find the first rejection reason
                rejection_reason = next(
                    (decision.get("reason", "Unknown rejection reason") for decision in claim_decisions if decision.get("status") == "rejected"),
                    "Multiple documents have validation issues",
                )
                claim_decision = ClaimDecision(status="rejected", reason=rejection_reason)
            else:
                # All decisions are approved but no good quality docs
                claim_decision = ClaimDecision(status="rejected", reason="No high-quality documents found")

    return ProcessClaimResponse(
        documents=documents,
        validation=validation,
        claim_decision=claim_decision,
    )
