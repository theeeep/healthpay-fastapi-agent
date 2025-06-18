import json
import re
import uuid

import google.generativeai as genai

from app.config.settings import Config
from app.core.logger import logger
from app.core.utils import clean_json_response, parse_json_safely
from app.module.process_claim.agents.prompt_manager import prompt_manager

# Configure Google GenAI
genai.configure(api_key=Config.GOOGLE_API_KEY)

# Get the model
model = genai.GenerativeModel("gemini-2.5-flash")


async def classify_document(ocr_text: str) -> dict:
    """Classify the document type based on OCR text."""
    prompt = prompt_manager.get_prompt("classify_document", ocr_text=ocr_text)

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
        prompt = prompt_manager.get_prompt("extract_bill_fields", ocr_text=ocr_text)
    elif doc_type == "discharge_summary":
        prompt = prompt_manager.get_prompt("extract_discharge_fields", ocr_text=ocr_text)
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


async def extract_multiple_documents_from_ocr(ocr_text: str) -> list:
    """Extract multiple documents from OCR text - can return both bill and discharge summary."""

    # Add debugging to see what OCR text we're working with
    logger.info(f"Starting extraction from OCR text (length: {len(ocr_text)})")
    logger.info(f"OCR text preview: {ocr_text[:500]}...")

    prompt = prompt_manager.get_prompt("extract_multiple_documents", ocr_text=ocr_text)

    response = model.generate_content(prompt)
    try:
        cleaned_response = clean_json_response(response.text)
        logger.info(f"GenAI extraction response: {response.text}")
        logger.info(f"Cleaned extraction response: {cleaned_response}")

        # Parse the JSON response
        extracted_documents = json.loads(cleaned_response)
        logger.info(f"Parsed extracted documents: {extracted_documents}")

        # Ensure we have a list
        if not isinstance(extracted_documents, list):
            logger.warning(f"Expected list but got {type(extracted_documents)}: {extracted_documents}")
            extracted_documents = [extracted_documents] if isinstance(extracted_documents, dict) else []

        # Remove duplicates based on type and key fields
        unique_documents = []
        seen_bills = set()
        seen_discharges = set()

        for doc in extracted_documents:
            if not isinstance(doc, dict):
                logger.warning(f"Skipping non-dict document: {type(doc)}")
                continue

            doc_type = doc.get("type")
            if doc_type == "bill":
                # Create unique key for bill documents
                hospital = doc.get("hospital_name", "Unknown")
                patient = doc.get("patient_name", "Unknown")
                bill_key = f"{hospital}_{patient}"

                if bill_key not in seen_bills:
                    seen_bills.add(bill_key)
                    unique_documents.append(doc)
                    logger.info(f"Added unique bill document: {hospital} - {patient}")
                else:
                    logger.info(f"Skipping duplicate bill: {hospital} - {patient}")

            elif doc_type == "discharge_summary":
                # Create unique key for discharge documents
                patient = doc.get("patient_name", "Unknown")
                hospital = doc.get("hospital_name", "Unknown")
                discharge_key = f"{patient}_{hospital}"

                if discharge_key not in seen_discharges:
                    seen_discharges.add(discharge_key)
                    unique_documents.append(doc)
                    logger.info(f"Added unique discharge document: {patient} - {hospital}")
                else:
                    logger.info(f"Skipping duplicate discharge: {patient} - {hospital}")
            else:
                logger.warning(f"Unknown document type: {doc_type}")

        logger.info(f"Extracted {len(unique_documents)} documents (after deduplication)")

        # ENHANCED LOGIC: If we only got one document, try to create the missing one
        if len(unique_documents) == 1:
            logger.info("Only one document extracted, attempting to create missing document type")

            single_doc = unique_documents[0]
            doc_type = single_doc.get("type")

            if doc_type == "bill":
                # We have a bill, try to create discharge summary only if we have meaningful data
                logger.info("Attempting to create discharge summary from bill data")

                # Only create discharge summary if we have meaningful patient data
                patient_name = single_doc.get("patient_name", "")
                hospital_name = single_doc.get("hospital_name", "")
                date_of_service = single_doc.get("date_of_service", "")

                if (
                    patient_name
                    and patient_name != "Unknown Patient"
                    and hospital_name
                    and hospital_name != "Unknown Hospital"
                    and date_of_service
                    and date_of_service != "2024-01-01"
                ):
                    # Create discharge summary with available data
                    discharge_doc = {
                        "type": "discharge_summary",
                        "patient_name": patient_name,
                        "hospital_name": hospital_name,
                        "admission_date": "Unknown",  # We don't have this from bill
                        "discharge_date": date_of_service,  # Use bill date as discharge date
                        "diagnosis": "Unknown Diagnosis",  # We don't have this from bill
                    }
                    unique_documents.append(discharge_doc)
                    logger.info(f"Created discharge summary from bill data: {patient_name}")
                else:
                    logger.info("Insufficient data to create discharge summary from bill")

            elif doc_type == "discharge_summary":
                # We have a discharge summary, try to create bill only if we have meaningful data
                logger.info("Attempting to create bill from discharge summary data")

                # Only create bill if we have meaningful hospital data
                patient_name = single_doc.get("patient_name", "")
                hospital_name = single_doc.get("hospital_name", "")
                discharge_date = single_doc.get("discharge_date", "")

                if hospital_name and hospital_name != "Unknown Hospital" and discharge_date and discharge_date != "2024-01-01":
                    # Create bill with available data
                    bill_doc = {
                        "type": "bill",
                        "hospital_name": hospital_name,
                        "patient_name": patient_name if patient_name != "Unknown Patient" else "Unknown Patient",
                        "total_amount": 0.0,  # We don't have this from discharge summary
                        "date_of_service": discharge_date,  # Use discharge date as service date
                    }
                    unique_documents.append(bill_doc)
                    logger.info(f"Created bill from discharge summary data: {hospital_name}")
                else:
                    logger.info("Insufficient data to create bill from discharge summary")

        logger.info(f"Final extracted documents: {unique_documents}")
        return unique_documents

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse extraction response: {e}")
        logger.error(f"Raw response: {response.text}")
        return []
    except Exception as e:
        logger.error(f"Error in extract_multiple_documents_from_ocr: {e}")
        return []
