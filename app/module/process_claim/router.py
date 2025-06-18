import uuid
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import ValidationError

from app.config.settings import Config
from app.core.logger import logger
from app.module.process_claim.agents.document_extractor import run_claim_processing_pipeline as run_genai_pipeline
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
    """Process medical insurance claim documents using AI-driven workflow."""
    # Validate number of files
    if len(files) > Config.MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files. Maximum {Config.MAX_FILES_PER_REQUEST} files allowed per request.",
        )

    ocr_texts = []

    # Validate and process uploaded files
    for file in files:
        # Validate file type
        if file.content_type not in Config.SUPPORTED_FILE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {file.filename}. Only PDF files are allowed.",
            )

        # Validate filename
        if file.filename is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is missing a filename.",
            )

        # Validate file size (rough estimation)
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > Config.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File {file.filename} is too large. Maximum {Config.MAX_FILE_SIZE_MB}MB allowed.",
            )

        ocr_text = await process_ocr(content, file.filename)
        ocr_texts.append(ocr_text)

    try:
        # Step 1: Use LLM (Gemini) for classification, extraction, and basic validation
        logger.info("Step 1: Using LLM (Gemini) for classification, extraction, and basic validation")
        genai_results = await run_genai_pipeline(ocr_texts, user_id=str(uuid.uuid4()))
        logger.info(f"GenAI pipeline returned {len(genai_results)} results")
        logger.info(f"GenAI results type: {type(genai_results)}")
        logger.info(f"GenAI results content: {genai_results}")

        # Extract the actual documents from GenAI results for ADK processing
        genai_extracted_documents = []
        for i, result in enumerate(genai_results):
            logger.info(f"Processing GenAI result {i}: type={type(result)}, content={result}")

            # Handle case where result might be a list
            if isinstance(result, list):
                logger.warning(f"GenAI result {i} is a list, processing each item")
                for item in result:
                    if isinstance(item, dict):
                        extracted_fields = item.get("extracted_fields")
                        if extracted_fields and isinstance(extracted_fields, dict):
                            genai_extracted_documents.append(extracted_fields)
                continue

            # Handle case where result is a dictionary
            if isinstance(result, dict):
                extracted_fields = result.get("extracted_fields")
                if extracted_fields and isinstance(extracted_fields, dict):
                    genai_extracted_documents.append(extracted_fields)
            else:
                logger.warning(f"GenAI result {i} is neither list nor dict: {type(result)}")

        logger.info(f"Extracted {len(genai_extracted_documents)} documents for ADK processing")

        # Step 2: Use ADK agents for enhanced validation and decision making (multi-agent orchestration)
        logger.info("Step 2: Using ADK agents for enhanced validation and decision making")
        adk_results = await run_adk_pipeline(genai_extracted_documents, user_id=str(uuid.uuid4()))
        logger.info(f"ADK pipeline returned {len(adk_results)} results")
        logger.info(f"ADK results type: {type(adk_results)}")
        logger.info(f"ADK results content: {adk_results}")

        # Combine results: Use GenAI for extraction, ADK for enhanced validation/decision
        agent_results = []

        # Add GenAI results (good extraction, basic validation, no decisions)
        for i, genai_result in enumerate(genai_results):
            logger.info(f"Adding GenAI result {i}: type={type(genai_result)}")

            # Handle case where genai_result might be a list
            if isinstance(genai_result, list):
                logger.info(f"GenAI result {i} is a list, adding each item")
                for item in genai_result:
                    if isinstance(item, dict):
                        agent_results.append(item)
            elif isinstance(genai_result, dict):
                agent_results.append(genai_result)
            else:
                logger.warning(f"Skipping GenAI result {i} with unexpected type: {type(genai_result)}")

        # Add ADK results (enhanced validation/decision) - but only if they provide value
        for i, adk_result in enumerate(adk_results):
            logger.info(f"Processing ADK result {i}: type={type(adk_result)}")
            logger.info(f"ADK result content: {adk_result}")

            # Ensure adk_result is a dictionary
            if not isinstance(adk_result, dict):
                logger.warning(f"ADK result {i} is not a dictionary: {type(adk_result)}, value: {adk_result}")
                continue

            # Only add ADK results if they have enhanced validation or decision making
            validation = adk_result.get("validation_result", {})
            decision = adk_result.get("claim_decision", {})

            logger.info(f"ADK validation: {validation}")
            logger.info(f"ADK decision: {decision}")

            # Always add ADK results if they have a claim decision, regardless of scores
            if (
                isinstance(decision, dict)
                and decision.get("status")  # Check for any claim decision
                or (isinstance(validation, dict) and validation.get("data_quality_score", 0) > 0)
                or (isinstance(decision, dict) and decision.get("confidence_score", 0) > 0)
                or (isinstance(validation, dict) and validation.get("recommendations"))
                or (isinstance(decision, dict) and decision.get("required_actions"))
            ):
                # ADK results should only contain validation and decision, not extracted fields
                adk_result["extracted_fields"] = None  # Ensure no extraction from ADK
                agent_results.append(adk_result)
                logger.info(f"Added ADK result {i} with decision: {decision.get('status', 'unknown') if isinstance(decision, dict) else 'unknown'}")
            else:
                logger.warning(f"Skipping ADK result {i} - no meaningful content found")
                logger.warning(f"Decision status: {decision.get('status') if isinstance(decision, dict) else 'not a dict'}")
                logger.warning(f"Data quality score: {validation.get('data_quality_score', 0) if isinstance(validation, dict) else 'not a dict'}")
                logger.warning(f"Confidence score: {decision.get('confidence_score', 0) if isinstance(decision, dict) else 0}")
                logger.warning(f"Has recommendations: {bool(validation.get('recommendations') if isinstance(validation, dict) else False)}")
                logger.warning(f"Has required actions: {bool(decision.get('required_actions') if isinstance(decision, dict) else False)}")

        logger.info(f"Combined {len(agent_results)} total results from both pipelines")

    except Exception as e:
        logger.error(f"Agent pipeline error: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal processing error") from e

    # Parse and validate agent output into Pydantic models
    documents = []
    all_missing_documents = []
    all_discrepancies = []
    claim_decisions = []

    logger.info(f"Processing {len(agent_results)} agent results")
    for i, result in enumerate(agent_results):
        # Handle case where result might be a string or have unexpected structure
        if isinstance(result, str):
            logger.warning(f"Agent returned string instead of dict: {result}")
            continue

        if not isinstance(result, dict):
            logger.warning(f"Agent returned unexpected type: {type(result)}")
            continue

        # Extract documents - assuming 'extracted_fields' contains document info
        extracted = result.get("extracted_fields", {})

        # Skip if no extracted fields (ADK pipeline only provides validation/decision)
        if extracted is None:
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

        # Aggregate validation results - prioritize ADK over GenAI
        validation_result = result.get("validation_result", {})
        if isinstance(validation_result, dict):
            missing_docs = validation_result.get("missing_documents", [])
            discrepancies = validation_result.get("discrepancies", [])

            # Only add GenAI validation if it's not overridden by ADK
            if isinstance(missing_docs, list):
                all_missing_documents.extend(missing_docs)
            if isinstance(discrepancies, list):
                all_discrepancies.extend(discrepancies)

        # Collect claim decisions - only ADK decisions (not GenAI pending)
        claim_decision = result.get("claim_decision", {})
        if isinstance(claim_decision, dict) and claim_decision.get("status") != "pending":
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

    # Determine final claim decision based on ADK results only
    adk_claim_decisions = []
    logger.info(f"Looking for ADK claim decisions in {len(agent_results)} agent results")

    for i, result in enumerate(agent_results):
        logger.info(f"Checking agent result {i}: {result}")
        claim_decision = result.get("claim_decision", {})
        logger.info(f"Claim decision in result {i}: {claim_decision}")

        if isinstance(claim_decision, dict):
            # Only use ADK decisions (not GenAI pending decisions)
            if claim_decision.get("status") != "pending":
                adk_claim_decisions.append(claim_decision)
                logger.info(f"Added ADK claim decision: {claim_decision}")
            else:
                logger.info(f"Skipping pending decision: {claim_decision}")
        else:
            logger.warning(f"Claim decision is not a dict: {type(claim_decision)}")

    logger.info(f"Found {len(adk_claim_decisions)} ADK claim decisions")

    if not adk_claim_decisions:
        claim_decision = ClaimDecision(status="rejected", reason="No ADK claim decision returned")
        logger.warning("No ADK claim decisions found - using fallback")
    else:
        # Use the first ADK decision (they should all be consistent)
        adk_decision = adk_claim_decisions[0]
        claim_decision = ClaimDecision(status=adk_decision.get("status", "rejected"), reason=adk_decision.get("reason", "Unknown reason"))
        logger.info(f"Using ADK decision: {claim_decision}")

    return ProcessClaimResponse(
        documents=documents,
        validation=validation,
        claim_decision=claim_decision,
    )
