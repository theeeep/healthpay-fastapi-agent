import asyncio
import json
import re
import uuid

import google.generativeai as genai

from app.config.settings import Config
from app.core.logger import logger

# Configure Google GenAI
genai.configure(api_key=Config.GOOGLE_API_KEY)


# Get the model
model = genai.GenerativeModel("gemini-2.0-flash-exp")


def clean_json_response(response_text: str) -> str:
    """Clean JSON response that might be wrapped in markdown code blocks."""
    # Remove markdown code block formatting
    response_text = re.sub(r"^```json\s*", "", response_text)
    response_text = re.sub(r"\s*```$", "", response_text)
    response_text = response_text.strip()
    return response_text


async def classify_document(ocr_text: str) -> dict:
    """Classify the document type based on OCR text."""
    prompt = f"""
    Analyze the following OCR text and classify the document type. Return ONLY a JSON object with the 'type' field.

    Document types to choose from:
    - bill: Hospital bills, medical bills, invoices
    - discharge_summary: Discharge summaries, medical reports, patient summaries

    OCR Text:
    {ocr_text}

    Return ONLY JSON like: {{"type": "bill"}} or {{"type": "discharge_summary"}}
    """

    response = model.generate_content(prompt)
    try:
        cleaned_response = clean_json_response(response.text)
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse classification response: {response.text}")
        return {"type": "unknown"}


async def extract_fields(ocr_text: str, doc_type: str) -> dict:
    """Extract relevant fields based on document type."""
    if doc_type == "bill":
        prompt = f"""
        Extract billing information from this OCR text. Return ONLY a JSON object with these fields:
        - type: "bill"
        - hospital_name: Name of the hospital (look for "GSTIN:" patterns, hospital names, or infer from context)
        - total_amount: Total amount as a number (look for amounts, bill numbers, or any large numbers)
        - date_of_service: Date in YYYY-MM-DD format (look for "From Date:", "ToDate:", "Date:")

        Look for these specific patterns:
        - "GSTIN: 36AAACA5443N3ZH" suggests Apollo Hospitals
        - "Bill No: INT1737245" or similar bill numbers
        - "From Date: 1-Feb-2025 ToDate: 2-Feb-2025" → extract dates
        - "Name: Mr. KOSGI VISHNUVARDHAN" → this is patient info, not hospital name
        - Look for any amounts, totals, or bill amounts in the text

        OCR Text:
        {ocr_text}

        Return ONLY JSON with extracted values. If you can't find a value, use defaults:
        - hospital_name: "Unknown Hospital" if not found
        - total_amount: 0.0 if not found  
        - date_of_service: "2024-01-01" if not found
        """
    elif doc_type == "discharge_summary":
        prompt = f"""
        Extract patient information from this discharge summary. Return ONLY a JSON object with these fields:
        - type: "discharge_summary"
        - patient_name: Name of the patient (look for "Name:" patterns)
        - diagnosis: Medical diagnosis (look for medical terms, conditions, procedures)
        - admission_date: Admission date in YYYY-MM-DD format
        - discharge_date: Discharge date in YYYY-MM-DD format

        Look for these specific patterns:
        - "Name: Mr. KOSGI VISHNUVARDHAN" → extract patient name
        - Look for medical conditions, diagnoses, or procedures
        - Look for admission and discharge dates
        - Look for medical specialty or department information

        OCR Text:
        {ocr_text}

        Return ONLY JSON with extracted values. If you can't find a value, use defaults:
        - patient_name: "Unknown Patient" if not found
        - diagnosis: "Unknown Diagnosis" if not found
        - admission_date: "2024-01-01" if not found
        - discharge_date: "2024-01-01" if not found
        """
    else:
        return {"type": "unknown"}

    response = model.generate_content(prompt)
    try:
        cleaned_response = clean_json_response(response.text)
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse extraction response: {response.text}")
        return {"type": doc_type}


async def validate_data(extracted_data: dict) -> dict:
    """Validate the extracted data for completeness."""
    prompt = f"""
    Validate this extracted data for completeness and consistency. Return ONLY a JSON object with:
    - missing_documents: List of missing required documents
    - discrepancies: List of any data inconsistencies or issues

    Extracted Data:
    {json.dumps(extracted_data, indent=2)}

    Return ONLY JSON like: {{"missing_documents": [], "discrepancies": []}}
    """

    response = model.generate_content(prompt)
    try:
        cleaned_response = clean_json_response(response.text)
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse validation response: {response.text}")
        return {"missing_documents": [], "discrepancies": ["Failed to validate data"]}


async def make_claim_decision(validation_result: dict) -> dict:
    """Make a claim decision based on validation results."""
    prompt = f"""
    Based on the validation results, make a claim decision. Return ONLY a JSON object with:
    - status: "approved" or "rejected"
    - reason: Brief explanation of the decision

    Validation Results:
    {json.dumps(validation_result, indent=2)}

    Return ONLY JSON like: {{"status": "approved", "reason": "All required documents present"}}
    """

    response = model.generate_content(prompt)
    try:
        cleaned_response = clean_json_response(response.text)
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse decision response: {response.text}")
        return {"status": "rejected", "reason": "Failed to process claim decision"}


async def run_claim_processing_pipeline(ocr_texts: list, user_id: str = None):
    """Run the complete claim processing pipeline."""
    user_id = user_id or str(uuid.uuid4())
    final_results = []

    try:
        for i, ocr_text in enumerate(ocr_texts):
            logger.info(f"=== GenAI Pipeline Processing Document {i + 1} ===")
            logger.info(f"GenAI OCR Text (first 500 chars): {ocr_text[:500]}...")

            # Step 1: Classify document
            classification = await classify_document(ocr_text)
            logger.info(f"GenAI Classification result: {classification}")

            # Step 2: Extract fields
            extracted_fields = await extract_fields(ocr_text, classification.get("type", "unknown"))
            logger.info(f"GenAI Extracted fields: {extracted_fields}")

            # Step 3: Validate data
            validation_result = await validate_data(extracted_fields)
            logger.info(f"GenAI Validation result: {validation_result}")

            # Step 4: Make claim decision
            claim_decision = await make_claim_decision(validation_result)
            logger.info(f"GenAI Claim decision: {claim_decision}")

            # Combine all results
            result = {"extracted_fields": extracted_fields, "validation_result": validation_result, "claim_decision": claim_decision}

            final_results.append(result)

    except Exception as e:
        logger.error(f"Error running GenAI claim processing pipeline: {e}")
        raise

    logger.info(f"GenAI Pipeline completed with {len(final_results)} results")
    return final_results
