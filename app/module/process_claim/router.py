"""
Process Claim Router - FastAPI router for claim processing endpoint.
This router handles HTTP concerns and delegates business logic to services.
"""

from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.logger import logger
from app.module.process_claim.schemas.schemas import ProcessClaimResponse
from app.module.process_claim.services.claim_processor import ClaimProcessor, ProcessingError

process_claim_router = APIRouter()


@process_claim_router.post("/process-claim", response_model=ProcessClaimResponse)
async def process_claim_documents(files: List[UploadFile] = File(...)):
    """
    Process medical insurance claim documents using AI-driven workflow.

    This endpoint:
    1. Accepts multiple PDF files (bill, discharge summary, etc.)
    2. Uses AI agents to classify and extract information
    3. Validates the extracted data
    4. Returns a claim decision with reasons

    Args:
        files: List of uploaded PDF files

    Returns:
        ProcessClaimResponse with processed documents and claim decision

    Raises:
        HTTPException: If processing fails or validation errors occur
    """
    try:
        # Extract file contents and names
        file_contents = []
        filenames = []

        for file in files:
            content = await file.read()
            file_contents.append(content)
            filenames.append(file.filename or "unknown.pdf")

        # Process claims using the service layer
        processor = ClaimProcessor()
        result = await processor.process_claim_documents(files=file_contents, filenames=filenames)

        logger.info(f"Successfully processed {len(files)} files")
        return result

    except ProcessingError as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during claim processing: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during claim processing")
