import json
import re
import uuid

import google.generativeai as genai

from app.config.settings import Config
from app.core.logger import logger
from app.core.utils import clean_json_response, parse_json_safely

# Configure Google GenAI
genai.configure(api_key=Config.GOOGLE_API_KEY)


# Get the model
model = genai.GenerativeModel("gemini-2.5-flash")


async def classify_document(ocr_text: str) -> dict:
    """Classify the document type based on OCR text."""
    prompt = f"""
    Analyze the following OCR text and classify the document type. Return ONLY a JSON object with the 'type' field.

    Document types to choose from:
    - bill: Hospital bills, medical bills, invoices, covering letters with bill amounts, final bill summaries
    - discharge_summary: Discharge summaries, medical reports, patient summaries, inpatient summaries

    Look for these key indicators:
    - BILL indicators: "Bill", "Invoice", "Amount", "Total", "Payable", "GST", 
      "Final Bill Summary", "Covering Letter", "Bill No"
    - DISCHARGE indicators: "Discharge Summary", "Inpatient Summary", "Patient Name", 
      "Diagnosis", "Admission", "Discharge Date", "Department of", "Admitted on", 
      "Discharged On", "Treating Doctor"

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


async def run_claim_processing_pipeline(ocr_texts: list, user_id: str = None):
    """Run the complete claim processing pipeline."""
    user_id = user_id or str(uuid.uuid4())
    final_results = []

    try:
        for i, ocr_text in enumerate(ocr_texts):
            logger.info(f"=== GenAI Pipeline Processing Document {i + 1} ===")

            # Step 1: Extract multiple documents from OCR
            extracted_documents = await extract_multiple_documents_from_ocr(ocr_text)
            logger.info(f"GenAI Extracted {len(extracted_documents)} documents")

            # Step 2: Process each extracted document (extraction only, no validation)
            for j, doc in enumerate(extracted_documents):
                # GenAI only extracts, ADK handles validation and decisions
                validation_result = {"missing_documents": [], "discrepancies": []}  # Empty validation from GenAI
                claim_decision = {"status": "pending", "reason": "Decision pending ADK processing"}

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

    # Add debugging to see what OCR text we're working with
    logger.info(f"Starting extraction from OCR text (length: {len(ocr_text)})")
    logger.info(f"OCR text preview: {ocr_text[:500]}...")

    prompt = f"""
    Analyze this OCR text and extract ALL possible documents. A single PDF can contain multiple document types.
    
    IMPORTANT: This PDF likely contains BOTH billing information AND discharge summary information.
    You MUST extract BOTH document types if both are present.
    
    Extract these fields if available:
    - patient_name: Name of the patient
    - hospital_name: Name of the hospital
    - total_amount: Use the TOTAL bill amount (not individual payable/non-payable amounts)
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
    - Amounts: Look for TOTAL bill amounts, not individual line items
    - Medical info: Department names, doctor names, diagnoses
    
    OCR Text:
    {ocr_text}
    
    CRITICAL: Return ONLY a JSON array of documents. Do not include any explanatory text, 
    markdown formatting, or additional text before or after the JSON.
    
    Each document should have a "type" field and appropriate fields:
    
    For BILL documents (ONLY ONE per patient/hospital):
    {{
      "type": "bill",
      "hospital_name": "Hospital Name",
      "total_amount": 12345.67,  // Use TOTAL amount, not individual amounts
      "date_of_service": "2025-02-11",
      "patient_name": "Patient Name"
    }}
    
    For DISCHARGE SUMMARY documents:
    {{
      "type": "discharge_summary", 
      "patient_name": "Patient Name",
      "diagnosis": "Medical Diagnosis",
      "admission_date": "2025-02-07",
      "discharge_date": "2025-02-11",
      "hospital_name": "Hospital Name"
    }}
    
    RULES:
    1. Create only ONE bill document per patient/hospital combination
    2. Use the TOTAL bill amount, not individual payable/non-payable amounts
    3. If the document contains both billing and discharge information, return BOTH document types
    4. Do not create multiple bill documents for the same patient
    5. IMPORTANT: If you see patient name, admission date, discharge date, and diagnosis - create a discharge_summary document
    6. IMPORTANT: If you see hospital name, total amount, and billing information - create a bill document
    7. A single PDF can and should return BOTH document types if both types of information are present
    
    EXAMPLES:
    - If you see "Patient Name: Mrs. Mary Philo", "Admission Date: 2025-02-07", "Discharge Date: 2025-02-11", "Diagnosis: LEFT KNEE INFECTED OSTEOARTHRITIS" → Create discharge_summary
    - If you see "FORTIS HOSPITALS LIMITED", "Total Amount: 435639.15", "Bill Date: 2025-02-11" → Create bill
    - If you see BOTH types of information → Create BOTH documents
    
    IMPORTANT: Return ONLY the JSON array, nothing else.
    """

    response = model.generate_content(prompt)

    # Add debugging for the response
    logger.info(f"GenAI extraction response: {response.text}")

    try:
        cleaned_response = clean_json_response(response.text)
        logger.info(f"Cleaned response: {cleaned_response}")

        documents = json.loads(cleaned_response)
        logger.info(f"Parsed documents: {documents}")

        # Ensure it's a list
        if isinstance(documents, dict):
            documents = [documents]
            logger.info(f"Converted single document to list: {documents}")

        # Post-process to remove duplicate bills for the same patient/hospital
        unique_documents = []
        seen_bills = set()

        for doc in documents:
            logger.info(f"Processing document: {doc}")
            if doc.get("type") == "bill":
                # Create a key for bill uniqueness
                bill_key = f"{doc.get('patient_name', '')}_{doc.get('hospital_name', '')}"
                if bill_key not in seen_bills:
                    seen_bills.add(bill_key)
                    unique_documents.append(doc)
                    logger.info(f"Added bill document: {doc}")
                else:
                    logger.info(f"Skipping duplicate bill for {bill_key}")
            else:
                # Non-bill documents (discharge summaries) are always unique
                unique_documents.append(doc)
                logger.info(f"Added non-bill document: {doc}")

        # ENHANCED LOGIC: If we only got one document, try to create the missing one
        if len(unique_documents) == 1:
            logger.info("Only one document extracted, attempting to create missing document type")

            single_doc = unique_documents[0]
            doc_type = single_doc.get("type")

            if doc_type == "bill":
                # We have a bill, try to create discharge summary only if we have meaningful data
                logger.info("Attempting to create discharge summary from bill data")

                # Only create discharge summary if we have meaningful patient and hospital data
                patient_name = single_doc.get("patient_name")
                hospital_name = single_doc.get("hospital_name")
                date_of_service = single_doc.get("date_of_service")

                if (
                    patient_name
                    and patient_name != "Unknown Patient"
                    and hospital_name
                    and hospital_name != "Unknown Hospital"
                    and date_of_service
                    and date_of_service != "2024-01-01"
                ):
                    # Try to extract additional data from OCR for discharge summary
                    discharge_summary = {
                        "type": "discharge_summary",
                        "patient_name": patient_name,
                        "diagnosis": "Unknown Diagnosis",  # Let validation catch this
                        "admission_date": "Unknown",  # Let validation catch this
                        "discharge_date": date_of_service,  # Use bill date as discharge date
                        "hospital_name": hospital_name,
                    }
                    unique_documents.append(discharge_summary)
                    logger.info(f"Added discharge summary with available data: {discharge_summary}")
                else:
                    logger.info("Insufficient data to create meaningful discharge summary - will be caught by validation")

            elif doc_type == "discharge_summary":
                # We have a discharge summary, try to create bill only if we have meaningful data
                logger.info("Attempting to create bill from discharge summary data")

                # Only create bill if we have meaningful patient and hospital data
                patient_name = single_doc.get("patient_name")
                hospital_name = single_doc.get("hospital_name")
                discharge_date = single_doc.get("discharge_date")

                if (
                    patient_name
                    and patient_name != "Unknown Patient"
                    and hospital_name
                    and hospital_name != "Unknown Hospital"
                    and discharge_date
                    and discharge_date != "2024-01-01"
                ):
                    # Try to extract additional data from OCR for bill
                    bill = {
                        "type": "bill",
                        "hospital_name": hospital_name,
                        "total_amount": 0.0,  # Let validation catch this
                        "date_of_service": discharge_date,  # Use discharge date as service date
                        "patient_name": patient_name,
                    }
                    unique_documents.append(bill)
                    logger.info(f"Added bill with available data: {bill}")
                else:
                    logger.info("Insufficient data to create meaningful bill - will be caught by validation")

        logger.info(f"Extracted {len(unique_documents)} documents (after deduplication and enhancement)")
        logger.info(f"Final documents: {unique_documents}")
        return unique_documents

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse extraction response: {response.text}")
        logger.error(f"Cleaned response: {cleaned_response}")
        logger.error(f"JSON decode error: {e}")

        # Try to extract JSON from the response more aggressively
        try:
            # Look for any JSON-like structure
            json_pattern = r"\[[\s\S]*\]|\{[\s\S]*\}"
            matches = re.findall(json_pattern, response.text)
            if matches:
                for match in matches:
                    try:
                        parsed = json.loads(match)
                        if isinstance(parsed, list):
                            logger.info(f"Recovered documents from pattern match: {parsed}")
                            return parsed
                        elif isinstance(parsed, dict):
                            logger.info(f"Recovered single document from pattern match: {parsed}")
                            return [parsed]
                    except json.JSONDecodeError:
                        continue
        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed: {recovery_error}")

        return []
