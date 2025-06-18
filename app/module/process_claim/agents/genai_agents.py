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
model = genai.GenerativeModel("gemini-2.5-flash")


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
    - bill: Hospital bills, medical bills, invoices, covering letters with bill amounts, final bill summaries
    - discharge_summary: Discharge summaries, medical reports, patient summaries, inpatient summaries

    Look for these key indicators:
    - BILL indicators: "Bill", "Invoice", "Amount", "Total", "Payable", "GST", "Final Bill Summary", "Covering Letter", "Bill No"
    - DISCHARGE indicators: "Discharge Summary", "Inpatient Summary", "Patient Name", "Diagnosis", "Admission", "Discharge Date", "Department of", "Admitted on", "Discharged On", "Treating Doctor"

    IMPORTANT CLASSIFICATION RULES:
    1. If the document contains BOTH bill information AND discharge information, classify based on PRIMARY content:
       - If it has patient details, admission/discharge dates, medical department, treating doctor → classify as "discharge_summary"
       - If it has amounts, bill numbers, payable amounts, GST details → classify as "bill"
    
    2. "INPATIENT SUMMARY RUNNING BILL" with patient details and medical info should be "discharge_summary"
    3. Look for medical context: department names, doctor names, admission/discharge dates
    4. If document has patient name, admission date, discharge date, department → it's likely a discharge_summary

    OCR Text:
    {ocr_text}

    Return ONLY JSON like: {{"type": "bill"}} or {{"type": "discharge_summary"}}
    """

    response = model.generate_content(prompt)
    try:
        cleaned_response = clean_json_response(response.text)
        result = json.loads(cleaned_response)
        logger.info(f"Classification prompt response: {response.text}")
        logger.info(f"Cleaned classification result: {result}")
        return result
    except json.JSONDecodeError:
        logger.error(f"Failed to parse classification response: {response.text}")
        return {"type": "unknown"}


async def extract_fields(ocr_text: str, doc_type: str) -> dict:
    """Extract relevant fields based on document type."""
    if doc_type == "bill":
        prompt = f"""
        Extract billing information from this OCR text. Return ONLY a JSON object with these fields:
        - type: "bill"
        - hospital_name: Name of the hospital (look for hospital names, GST numbers, or infer from context)
        - total_amount: Total amount as a number (look for amounts, bill numbers, or any large numbers)
        - date_of_service: Date in YYYY-MM-DD format (look for dates, bill dates, service dates)
        - patient_name: Name of the patient if available (look for "Patient Name:", "Name:")

        Look for these specific patterns:
        - Hospital names: "FORTIS HOSPITALS LIMITED", "Max Super Specialty Hospital", "SIR GANGA RAM HOSPITAL"
        - GST numbers: "GSTIN:", "GST No."
        - Amounts: Look for large numbers that could be bill amounts
        - Dates: Look for dates in various formats and convert to YYYY-MM-DD
        - Patient names: "Patient Name:", "Name:"

        OCR Text:
        {ocr_text}

        Return ONLY JSON with extracted values. If you can't find a value, use defaults:
        - hospital_name: "Unknown Hospital" if not found
        - total_amount: 0.0 if not found  
        - date_of_service: "2024-01-01" if not found
        - patient_name: "Unknown Patient" if not found
        """
    elif doc_type == "discharge_summary":
        prompt = f"""
        Extract patient information from this discharge summary. Return ONLY a JSON object with these fields:
        - type: "discharge_summary"
        - patient_name: Name of the patient (look for "Patient Name:" patterns)
        - diagnosis: Medical diagnosis (look for medical terms, conditions, procedures)
        - admission_date: Admission date in YYYY-MM-DD format
        - discharge_date: Discharge date in YYYY-MM-DD format
        - hospital_name: Name of the hospital if available (look for hospital names, GST numbers)

        Look for these specific patterns:
        - Patient names: "Patient Name:", "Name:"
        - Medical conditions: Look for medical terms, diagnoses, or procedures
        - Dates: "Date of Admission", "Date Of Discharge", "Admitted on", "Discharged On"
        - Hospital names: Look for hospital names, GST numbers, or department information

        OCR Text:
        {ocr_text}

        Return ONLY JSON with extracted values. If you can't find a value, use defaults:
        - patient_name: "Unknown Patient" if not found
        - diagnosis: "Unknown Diagnosis" if not found
        - admission_date: "2024-01-01" if not found
        - discharge_date: "2024-01-01" if not found
        - hospital_name: "Unknown Hospital" if not found
        """
    else:
        return {"type": "unknown"}

    response = model.generate_content(prompt)
    try:
        cleaned_response = clean_json_response(response.text)
        result = json.loads(cleaned_response)
        logger.info(f"Extraction prompt response: {response.text}")
        logger.info(f"Cleaned extraction result: {result}")
        return result
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
            logger.info(f"GenAI OCR Text length: {len(ocr_text)} characters")
            logger.info(f"GenAI OCR Text (first 1000 chars): {ocr_text[:1000]}...")

            # Step 1: Extract multiple documents from OCR
            extracted_documents = await extract_multiple_documents_from_ocr(ocr_text)
            logger.info(f"GenAI Extracted {len(extracted_documents)} documents: {extracted_documents}")

            # Step 2: Process each extracted document
            for j, doc in enumerate(extracted_documents):
                logger.info(f"Processing extracted document {j + 1}/{len(extracted_documents)}: {doc.get('type', 'unknown')}")

                # Validate the document data
                validation_result = await validate_data(doc)
                logger.info(f"GenAI Validation result for {doc.get('type')}: {validation_result}")

                # Make claim decision for this document
                claim_decision = await make_claim_decision(validation_result)
                logger.info(f"GenAI Claim decision for {doc.get('type')}: {claim_decision}")

                # Combine results for this document
                result = {"extracted_fields": doc, "validation_result": validation_result, "claim_decision": claim_decision}
                final_results.append(result)

    except Exception as e:
        logger.error(f"Error running GenAI claim processing pipeline: {e}")
        raise

    logger.info(f"GenAI Pipeline completed with {len(final_results)} total document results")
    return final_results


async def extract_all_fields_from_ocr(ocr_text: str) -> dict:
    """Extract all possible fields from OCR text, then classify and structure."""
    prompt = f"""
    Extract ALL possible information from this OCR text and return a structured JSON object.
    
    Extract these fields if available:
    - patient_name: Name of the patient
    - hospital_name: Name of the hospital
    - total_amount: Any amount or bill amount
    - date_of_service: Any service date, bill date
    - admission_date: Admission date
    - discharge_date: Discharge date
    - diagnosis: Medical diagnosis or condition
    - department: Medical department
    - doctor: Treating doctor name
    
    Look for these patterns:
    - Patient names: "Patient Name:", "Name:"
    - Hospital names: Look for hospital names, GST numbers
    - Dates: "Admitted on", "Discharged On", "Date:", "Bill Date"
    - Amounts: Look for large numbers that could be amounts
    - Medical info: Department names, doctor names, diagnoses
    
    OCR Text:
    {ocr_text}
    
    Return ONLY JSON with all extracted values. Use "Unknown" for missing fields.
    """

    response = model.generate_content(prompt)
    try:
        cleaned_response = clean_json_response(response.text)
        all_fields = json.loads(cleaned_response)
        logger.info(f"All fields extracted: {all_fields}")
        return all_fields
    except json.JSONDecodeError:
        logger.error(f"Failed to parse extraction response: {response.text}")
        return {}


async def classify_and_structure_document(all_fields: dict) -> dict:
    """Classify document type and structure data appropriately."""
    # Determine document type based on available fields
    has_billing_info = any([all_fields.get("total_amount", 0) > 0, "bill" in str(all_fields).lower(), "amount" in str(all_fields).lower()])

    has_discharge_info = any(
        [
            all_fields.get("admission_date"),
            all_fields.get("discharge_date"),
            all_fields.get("diagnosis"),
            all_fields.get("department"),
            "discharge" in str(all_fields).lower(),
            "inpatient" in str(all_fields).lower(),
        ]
    )

    # Classification logic
    if has_discharge_info and (has_billing_info is False or has_discharge_info is True):
        # If it has discharge info and either no billing info OR strong discharge indicators
        doc_type = "discharge_summary"
        structured_data = {
            "type": "discharge_summary",
            "patient_name": all_fields.get("patient_name", "Unknown Patient"),
            "diagnosis": all_fields.get("diagnosis", "Unknown Diagnosis"),
            "admission_date": all_fields.get("admission_date", "2024-01-01"),
            "discharge_date": all_fields.get("discharge_date", "2024-01-01"),
            "hospital_name": all_fields.get("hospital_name", "Unknown Hospital"),
        }
    else:
        # Default to bill if no clear discharge indicators
        doc_type = "bill"
        structured_data = {
            "type": "bill",
            "hospital_name": all_fields.get("hospital_name", "Unknown Hospital"),
            "total_amount": all_fields.get("total_amount", 0.0),
            "date_of_service": all_fields.get("date_of_service", "2024-01-01"),
            "patient_name": all_fields.get("patient_name", "Unknown Patient"),
        }

    logger.info(f"Classified as: {doc_type}")
    logger.info(f"Structured data: {structured_data}")
    return structured_data


async def extract_multiple_documents_from_ocr(ocr_text: str) -> list:
    """Extract multiple documents from OCR text - can return both bill and discharge summary."""
    prompt = f"""
    Analyze this OCR text and extract ALL possible documents. A single PDF can contain multiple document types.
    
    Extract these fields if available:
    - patient_name: Name of the patient
    - hospital_name: Name of the hospital
    - total_amount: Any amount or bill amount
    - date_of_service: Any service date, bill date
    - admission_date: Admission date
    - discharge_date: Discharge date
    - diagnosis: Medical diagnosis or condition
    - department: Medical department
    - doctor: Treating doctor name
    
    Look for these patterns:
    - Patient names: "Patient Name:", "Name:"
    - Hospital names: Look for hospital names, GST numbers
    - Dates: "Admitted on", "Discharged On", "Date:", "Bill Date"
    - Amounts: Look for large numbers that could be amounts
    - Medical info: Department names, doctor names, diagnoses
    
    OCR Text:
    {ocr_text}
    
    Return a JSON array of documents. Each document should have a "type" field and appropriate fields:
    
    For BILL documents:
    {{
      "type": "bill",
      "hospital_name": "Hospital Name",
      "total_amount": 12345.67,
      "date_of_service": "2025-02-11"
    }}
    
    For DISCHARGE SUMMARY documents:
    {{
      "type": "discharge_summary", 
      "patient_name": "Patient Name",
      "diagnosis": "Medical Diagnosis",
      "admission_date": "2025-02-07",
      "discharge_date": "2025-02-11"
    }}
    
    If the document contains both billing and discharge information, return BOTH document types.
    """

    response = model.generate_content(prompt)
    try:
        cleaned_response = clean_json_response(response.text)
        documents = json.loads(cleaned_response)

        # Ensure it's a list
        if isinstance(documents, dict):
            documents = [documents]

        logger.info(f"Extracted documents: {documents}")
        return documents
    except json.JSONDecodeError:
        logger.error(f"Failed to parse extraction response: {response.text}")
        return []
