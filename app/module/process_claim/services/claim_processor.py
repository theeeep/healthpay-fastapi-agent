"""
Claim Processing Service - Core business logic for processing medical insurance claims.
This service orchestrates the entire claim processing workflow in a clean, modular way.
"""

import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.core.logger import logger
from app.module.process_claim.agents.adk_validation_agent import run_claim_processing_pipeline as run_adk_pipeline
from app.module.process_claim.agents.genai_extraction_agent import run_claim_processing_pipeline as run_genai_pipeline
from app.module.process_claim.schemas.schemas import (
    BillDocument,
    ClaimDecision,
    DischargeSummaryDocument,
    ProcessClaimResponse,
    ValidationResult,
)
from app.module.process_claim.services.file_validator import FileValidator
from app.module.process_claim.services.mistral_ocr_service import process_ocr


@dataclass
class ProcessingResult:
    """Result of claim processing with all intermediate data."""

    documents: List[Dict]
    validation: ValidationResult
    claim_decision: ClaimDecision
    processing_metadata: Dict


class ClaimProcessor:
    """
    Core service for processing medical insurance claims.

    This service orchestrates the entire workflow:
    1. File validation and OCR processing
    2. Document extraction using GenAI
    3. Validation and decision making using ADK agents
    4. Response formatting and validation
    """

    def __init__(self):
        self.file_validator = FileValidator()

    async def process_claim_documents(self, files: List[bytes], filenames: List[str], user_id: Optional[str] = None) -> ProcessClaimResponse:
        """
        Process medical insurance claim documents using AI-driven workflow.

        Args:
            files: List of file contents as bytes
            filenames: List of corresponding filenames
            user_id: Optional user ID for tracking

        Returns:
            ProcessClaimResponse with processed documents and decisions

        Raises:
            ValueError: If files and filenames don't match
            ProcessingError: If processing fails
        """
        if len(files) != len(filenames):
            raise ValueError("Files and filenames lists must have the same length")

        user_id = user_id or str(uuid.uuid4())
        logger.info(f"Starting claim processing for user: {user_id}")

        try:
            # Step 1: Validate and process files
            ocr_results = await self._process_files(files, filenames)

            # Step 2: Extract documents using GenAI
            genai_results = await self._extract_documents(ocr_results, user_id)

            # Step 3: Validate and make decisions using ADK
            adk_results = await self._validate_and_decide(genai_results, user_id)

            # Step 4: Combine and format results
            result = await self._combine_results(genai_results, adk_results)

            logger.info(f"Claim processing completed successfully for user: {user_id}")
            return result

        except Exception as e:
            logger.error(f"Claim processing failed for user {user_id}: {e}")
            raise ProcessingError(f"Failed to process claim documents: {e}") from e

    async def _process_files(self, files: List[bytes], filenames: List[str]) -> List[Dict[str, str]]:
        """Validate files and extract OCR text."""
        ocr_results = []

        for file_content, filename in zip(files, filenames):
            # Validate file
            await self.file_validator.validate_file(file_content, filename)

            # Extract OCR text
            ocr_text = await process_ocr(file_content, filename)
            ocr_results.append({"text": ocr_text, "filename": filename})

        logger.info(f"Processed {len(ocr_results)} files with OCR")
        return ocr_results

    async def _extract_documents(self, ocr_results: List[Dict[str, str]], user_id: str) -> List[Dict]:
        """Extract documents using GenAI pipeline."""
        logger.info("Starting document extraction with GenAI")

        try:
            genai_results = await run_genai_pipeline(ocr_results, user_id=user_id)
            logger.info(f"GenAI extracted {len(genai_results)} document results")
            return genai_results
        except Exception as e:
            logger.error(f"GenAI extraction failed: {e}")
            raise ProcessingError(f"Document extraction failed: {e}") from e

    async def _validate_and_decide(self, genai_results: List[Dict], user_id: str) -> List[Dict]:
        """Validate documents and make decisions using ADK agents."""
        logger.info("Starting validation and decision making with ADK")

        try:
            # Extract documents for ADK processing
            extracted_documents = self._extract_documents_for_adk(genai_results)

            # Run ADK pipeline
            adk_results = await run_adk_pipeline(extracted_documents, user_id=user_id)
            logger.info(f"ADK processed {len(adk_results)} results")
            return adk_results
        except Exception as e:
            logger.error(f"ADK processing failed: {e}")
            raise ProcessingError(f"Validation and decision making failed: {e}") from e

    def _extract_documents_for_adk(self, genai_results: List[Dict]) -> List[Dict]:
        """Extract document data from GenAI results for ADK processing."""
        extracted_documents = []

        for result in genai_results:
            if isinstance(result, dict):
                extracted_fields = result.get("extracted_fields")
                if extracted_fields and isinstance(extracted_fields, dict):
                    extracted_documents.append(extracted_fields)
            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        extracted_fields = item.get("extracted_fields")
                        if extracted_fields and isinstance(extracted_fields, dict):
                            extracted_documents.append(extracted_fields)

        logger.info(f"Extracted {len(extracted_documents)} documents for ADK processing")
        return extracted_documents

    async def _combine_results(self, genai_results: List[Dict], adk_results: List[Dict]) -> ProcessClaimResponse:
        """Combine GenAI and ADK results into final response."""
        logger.info("Combining results from both pipelines")

        # Process documents from GenAI results
        documents = self._process_documents(genai_results)

        # Process validation and decisions from ADK results
        validation, claim_decision = self._process_validation_and_decisions(adk_results)

        return ProcessClaimResponse(
            documents=documents,
            validation=validation,
            claim_decision=claim_decision,
        )

    def _process_documents(self, genai_results: List[Dict]) -> List[Dict]:
        """Process and validate documents from GenAI results."""
        documents = []

        for result in genai_results:
            if isinstance(result, dict):
                extracted_fields = result.get("extracted_fields")
                if extracted_fields and isinstance(extracted_fields, dict):
                    try:
                        if extracted_fields.get("type") == "bill":
                            doc = BillDocument(**extracted_fields)
                            documents.append(doc)
                        elif extracted_fields.get("type") == "discharge_summary":
                            doc = DischargeSummaryDocument(**extracted_fields)
                            documents.append(doc)
                    except Exception as e:
                        logger.warning(f"Failed to create document from {extracted_fields}: {e}")

        logger.info(f"Processed {len(documents)} valid documents")
        return documents

    def _process_validation_and_decisions(self, adk_results: List[Dict]) -> tuple[ValidationResult, ClaimDecision]:
        """Process validation and decision results from ADK."""
        all_missing_documents = []
        all_discrepancies = []
        claim_decisions = []

        for result in adk_results:
            if isinstance(result, dict):
                # Process validation
                validation = result.get("validation_result", {})
                if isinstance(validation, dict):
                    missing_docs = validation.get("missing_documents", [])
                    discrepancies = validation.get("discrepancies", [])

                    if isinstance(missing_docs, list):
                        all_missing_documents.extend(missing_docs)
                    if isinstance(discrepancies, list):
                        all_discrepancies.extend(discrepancies)

                # Process decisions
                decision = result.get("claim_decision", {})
                if isinstance(decision, dict) and decision.get("status") != "pending":
                    claim_decisions.append(decision)

        # Create final validation result
        validation = ValidationResult(
            missing_documents=list(set(all_missing_documents)),  # Remove duplicates
            discrepancies=list(set(all_discrepancies)),  # Remove duplicates
        )

        # Create final claim decision
        if claim_decisions:
            # Use the first valid decision
            final_decision = claim_decisions[0]
            claim_decision = ClaimDecision(status=final_decision.get("status", "rejected"), reason=final_decision.get("reason", "Unknown reason"))
        else:
            claim_decision = ClaimDecision(status="rejected", reason="No valid claim decision returned")

        return validation, claim_decision


class ProcessingError(Exception):
    """Custom exception for claim processing errors."""

    pass
