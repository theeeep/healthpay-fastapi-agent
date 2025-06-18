import asyncio
import json
import re
import uuid
from typing import Dict, List

import dotenv
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.core.logger import logger

dotenv.load_dotenv()

# Configure session service
session_service = InMemorySessionService()


def clean_json_response(response_text: str) -> str:
    """Clean JSON response that might be wrapped in markdown code blocks."""
    # Remove markdown code block formatting
    response_text = re.sub(r"^```json\s*", "", response_text)
    response_text = re.sub(r"\s*```$", "", response_text)
    response_text = response_text.strip()
    return response_text


# Step 1: Document Classification Agent
classification_agent = LlmAgent(
    name="DocumentClassifier",
    model="gemini-2.0-flash-exp",
    description="Classifies document types from OCR text",
    instruction="""
    You are a document classification agent. Your task is to analyze OCR text and classify the document type.
    
    CRITICAL: You must return ONLY valid JSON. Do not include any markdown formatting, explanations, or additional text.
    
    Document types to classify:
    - "bill": Hospital bills, medical bills, invoices, billing statements, interim bills, final bills, GST bills
    - "discharge_summary": Discharge summaries, medical reports, patient summaries, clinical notes, medical records
    
    Look for these specific keywords to classify:
    
    BILL indicators (ANY of these indicate a bill):
    - "bill", "invoice", "interim bill", "final bill", "bill of supply"
    - "GSTIN", "GST", "tax", "amount", "total", "payment"
    - "bill to", "amount due", "total amount", "final amount"
    - "hospital bill", "medical bill", "patient bill"
    - "Interim Bill", "Bill Of Supply", "GSTIN:"
    
    DISCHARGE SUMMARY indicators (ALL of these should be present):
    - "discharge", "admission", "discharge summary", "medical summary"
    - "patient report", "clinical notes", "medical record"
    - "diagnosis", "treatment", "medication", "prescription"
    - "admission date", "discharge date", "length of stay"
    
    IMPORTANT: If you see "Interim Bill", "Bill Of Supply", "GSTIN:", or "Bill To" in the text, it's ALWAYS a bill.
    
    Return format: {"type": "bill"} or {"type": "discharge_summary"}
    
    Analyze the provided OCR text and return the appropriate JSON response.
    """,
    output_key="document_type",
)

# Step 2: Data Extraction Agent
extraction_agent = LlmAgent(
    name="DataExtractor",
    model="gemini-2.0-flash-exp",
    description="Extracts structured data from medical documents",
    instruction="""
    You are a data extraction agent. Extract structured information from medical documents based on document type.
    
    CRITICAL: You must return ONLY valid JSON. Do not include any markdown formatting, explanations, or additional text.
    
    First, determine the document type by analyzing the OCR text. Look for keywords:
    - BILL indicators: "bill", "invoice", "amount", "total", "payment", "hospital bill", "medical bill", "Interim Bill", "Bill Of Supply", "GSTIN"
    - DISCHARGE SUMMARY indicators: "discharge", "admission", "patient", "diagnosis", "medical report", "summary"
    
    Then extract ONLY the relevant fields based on the document type:
    
    FOR BILL DOCUMENTS (type: "bill"), extract ONLY:
    - type: "bill"
    - hospital_name: Extract the name of the hospital, clinic, or medical facility
    - total_amount: Extract the total bill amount as a number (float)
    - date_of_service: Extract the date of service in YYYY-MM-DD format
    
    FOR DISCHARGE SUMMARY DOCUMENTS (type: "discharge_summary"), extract ONLY:
    - type: "discharge_summary"
    - patient_name: Extract the patient's full name
    - diagnosis: Extract the medical diagnosis, condition, or reason for admission
    - admission_date: Extract the admission date in YYYY-MM-DD format
    - discharge_date: Extract the discharge date in YYYY-MM-DD format
    
    Look for these SPECIFIC patterns in the OCR text:
    
    FOR BILLS:
    HOSPITAL NAME patterns:
    - Look for hospital names like "Apollo Hospitals", "CARE Hospitals", "MAX Healthcare"
    - Look for medical center names
    - Look for healthcare provider names
    
    AMOUNT patterns:
    - Look for "Total:", "Amount Due:", "Bill Amount:", "Total Amount:", "Final Amount:"
    - Look for numbers followed by currency symbols or amounts
    - Extract the numeric value as a float
    
    DATE patterns:
    - "From Date: 1-Feb-2025 ToDate: 2-Feb-2025" → extract "2025-02-01" or "2025-02-02"
    - "Date : 02-Feb-2025" → extract "2025-02-02"
    - "Service Date: 2024/01/15" → extract "2024-01-15"
    - Convert any date format to YYYY-MM-DD
    
    FOR DISCHARGE SUMMARIES:
    PATIENT NAME patterns:
    - "Name: Mr. KOSGI VISHNUVARDHAN" → extract "Mr. KOSGI VISHNUVARDHAN"
    - "Patient: John Doe" → extract "John Doe"
    - "Name: Jane Smith" → extract "Jane Smith"
    - Look for "Name:" followed by a person's name
    
    DIAGNOSIS patterns:
    - Look for "Diagnosis:", "Condition:", "Medical Condition:"
    - Look for medical terms, conditions, or reasons for admission
    
    DATE patterns:
    - "Admit Date: 03/02/2025" → extract "2025-02-03"
    - "Admission Date: 2024-01-15" → extract "2024-01-15"
    - "Discharge Date: 2024-01-20" → extract "2024-01-20"
    - Convert any date format to YYYY-MM-DD
    
    IMPORTANT: You MUST extract all required fields for the document type. If you cannot find a field in the text:
    
    FOR BILLS:
    - For hospital_name: Use "Unknown Hospital" if no hospital name is found
    - For total_amount: Use 0.0 if no amount is found
    - For date_of_service: Use "2024-01-01" if no date is found
    
    FOR DISCHARGE SUMMARIES:
    - For patient_name: Use "Unknown Patient" if no patient name is found
    - For diagnosis: Use "Unknown Diagnosis" if no diagnosis is found
    - For admission_date: Use "2024-01-01" if no admission date is found
    - For discharge_date: Use "2024-01-01" if no discharge date is found
    
    Return format for BILL: {
        "type": "bill",
        "hospital_name": "Extracted Hospital Name or Unknown Hospital",
        "total_amount": 1234.56,
        "date_of_service": "2024-01-15"
    }
    
    Return format for DISCHARGE SUMMARY: {
        "type": "discharge_summary",
        "patient_name": "Extracted Patient Name or Unknown Patient",
        "diagnosis": "Extracted Diagnosis or Unknown Diagnosis",
        "admission_date": "2024-01-10",
        "discharge_date": "2024-01-15"
    }
    
    NEVER return null or empty strings for required fields. Always provide a value.
    NEVER mix fields from different document types.
    """,
    output_key="extracted_data",
)

# Step 3: Validation Agent
validation_agent = LlmAgent(
    name="DataValidator",
    model="gemini-2.0-flash-exp",
    description="Validates extracted data for completeness and consistency",
    instruction="""
    You are a data validation agent. Validate the extracted data for completeness and consistency.
    
    CRITICAL: You must return ONLY valid JSON. Do not include any markdown formatting, explanations, or additional text.
    
    Check for:
    1. Missing required document types (bill and/or discharge_summary)
    2. Missing required fields for each document type
    3. Data inconsistencies (dates, amounts, etc.)
    4. Invalid formats (dates, amounts, etc.)
    
    For BILL documents, check:
    - hospital_name: Should not be "Unknown Hospital"
    - total_amount: Should be greater than 0
    - date_of_service: Should be a valid date
    
    For DISCHARGE SUMMARY documents, check:
    - patient_name: Should not be "Unknown Patient"
    - diagnosis: Should not be "Unknown Diagnosis"
    - admission_date: Should be a valid date
    - discharge_date: Should be a valid date and after admission_date
    
    Missing document types to check:
    - If no bill document is found, add "bill" to missing_documents
    - If no discharge_summary document is found, add "discharge_summary" to missing_documents
    
    Common discrepancies to check:
    - "Total amount is zero" if total_amount is 0.0
    - "Hospital name is unknown" if hospital_name is "Unknown Hospital"
    - "Patient name is unknown" if patient_name is "Unknown Patient"
    - "Diagnosis is unknown" if diagnosis is "Unknown Diagnosis"
    - "Invalid date format" if dates are not in YYYY-MM-DD format
    - "Discharge date is before admission date" if discharge_date < admission_date
    - "Dates are in the future" if dates are in the future
    
    Return format: {
        "missing_documents": ["list of missing document types"],
        "discrepancies": ["list of data inconsistencies"]
    }
    
    Examples:
    - {"missing_documents": ["discharge_summary"], "discrepancies": ["Total amount is zero"]}
    - {"missing_documents": [], "discrepancies": ["Hospital name is unknown", "Patient name is unknown"]}
    - {"missing_documents": [], "discrepancies": []}
    """,
    output_key="validation_result",
)

# Step 4: Decision Agent
decision_agent = LlmAgent(
    name="ClaimDecisionMaker",
    model="gemini-2.0-flash-exp",
    description="Makes claim approval/rejection decisions based on validation results",
    instruction="""
    You are a claim decision agent. Make approve/reject decisions based on validation results.
    
    CRITICAL: You must return ONLY valid JSON. Do not include any markdown formatting, explanations, or additional text.
    
    Decision criteria:
    - APPROVE if no missing documents and no discrepancies
    - REJECT if there are missing documents or significant discrepancies
    
    Return format: {
        "status": "approved" or "rejected",
        "reason": "Brief explanation of the decision"
    }
    
    Examples:
    - {"status": "approved", "reason": "All required documents present and data is consistent"}
    - {"status": "rejected", "reason": "Missing required documents: patient ID, insurance card"}
    """,
    output_key="claim_decision",
)

# Create the sequential pipeline following ADK patterns
document_processing_pipeline = SequentialAgent(
    name="DocumentProcessingPipeline", sub_agents=[classification_agent, extraction_agent, validation_agent, decision_agent]
)


async def run_agent_pipeline(agent: LlmAgent, content: types.Content, user_id: str, session_id: str) -> Dict:
    """Run a single agent and return its response."""
    runner = Runner(agent=agent, app_name="healthpay_claims", session_service=session_service)

    # Log what the agent is receiving
    ocr_text = content.parts[0].text if content.parts else ""
    logger.info(f"Agent {agent.name} receiving input (first 300 chars): {ocr_text[:300]}...")

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            response_text = event.content.parts[0].text if event.content.parts else ""
            logger.info(f"Agent {agent.name} raw response: {response_text}")

            try:
                # Clean and parse JSON response
                cleaned_response = clean_json_response(response_text)
                parsed_response = json.loads(cleaned_response)
                logger.info(f"Agent {agent.name} parsed response: {parsed_response}")
                return parsed_response
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from {agent.name}: {response_text}")
                return {"error": f"Failed to parse response from {agent.name}"}

    return {"error": f"No response from {agent.name}"}


async def run_claim_processing_pipeline(ocr_texts: List[str], user_id: str = None) -> List[Dict]:
    """Run the complete multi-agent claim processing pipeline."""
    user_id = user_id or str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    # Create session
    await session_service.create_session(app_name="healthpay_claims", user_id=user_id, session_id=session_id)

    final_results = []

    try:
        for i, ocr_text in enumerate(ocr_texts):
            logger.info(f"=== Processing Document {i + 1} ===")
            logger.info(f"Original OCR Text (first 500 chars): {ocr_text[:500]}...")

            # Run the sequential pipeline
            content = types.Content(parts=[types.Part.from_text(text=ocr_text)])

            # Use the SequentialAgent pipeline
            pipeline_runner = Runner(agent=document_processing_pipeline, app_name="healthpay_claims", session_service=session_service)

            pipeline_result = {}
            async for event in pipeline_runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
                if event.is_final_response():
                    response_text = event.content.parts[0].text if event.content.parts else ""
                    logger.info(f"Pipeline raw response: {response_text}")
                    try:
                        cleaned_response = clean_json_response(response_text)
                        pipeline_result = json.loads(cleaned_response)
                        logger.info(f"Pipeline parsed result: {pipeline_result}")
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse pipeline response: {response_text}")
                        pipeline_result = {"error": "Failed to parse pipeline response"}

            # Check session state to see what each agent produced
            session = await session_service.get_session(app_name="healthpay_claims", user_id=user_id, session_id=session_id)
            logger.info(f"Session state after pipeline: {session.state}")

            # Extract data from session state
            extracted_data = {}
            validation_result = {}
            claim_decision = {}

            # Parse extracted_data from session state
            session_extracted_data = session.state.get("extracted_data", "")
            if session_extracted_data:
                if isinstance(session_extracted_data, str):
                    try:
                        cleaned_session_data = clean_json_response(session_extracted_data)
                        extracted_data = json.loads(cleaned_session_data)
                        logger.info(f"Parsed session extracted data: {extracted_data}")
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse session extracted data: {session_extracted_data}")
                        extracted_data = {}
                else:
                    extracted_data = session_extracted_data

            # Parse validation_result from session state
            session_validation = session.state.get("validation_result", "")
            if session_validation:
                if isinstance(session_validation, str):
                    try:
                        cleaned_validation = clean_json_response(session_validation)
                        validation_result = json.loads(cleaned_validation)
                    except json.JSONDecodeError:
                        validation_result = {"missing_documents": [], "discrepancies": ["Validation failed"]}
                else:
                    validation_result = session_validation

            # Parse claim_decision from session state
            session_decision = session.state.get("claim_decision", "")
            if session_decision:
                if isinstance(session_decision, str):
                    try:
                        cleaned_decision = clean_json_response(session_decision)
                        claim_decision = json.loads(cleaned_decision)
                    except json.JSONDecodeError:
                        claim_decision = {"status": "rejected", "reason": "Decision processing failed"}
                else:
                    claim_decision = session_decision

            # If no extracted_data, create a basic structure
            if not extracted_data:
                # Try to get document type from classification agent
                session_doc_type = session.state.get("document_type", "")
                doc_type = "unknown"
                if session_doc_type:
                    if isinstance(session_doc_type, str):
                        try:
                            cleaned_doc_type = clean_json_response(session_doc_type)
                            doc_type_data = json.loads(cleaned_doc_type)
                            doc_type = doc_type_data.get("type", "unknown")
                        except json.JSONDecodeError:
                            doc_type = "unknown"
                    elif isinstance(session_doc_type, dict):
                        doc_type = session_doc_type.get("type", "unknown")

                logger.info(f"Document type from session: {doc_type}")

                # Create a basic extracted_data structure
                extracted_data = {
                    "type": doc_type,
                    "hospital_name": "Unknown Hospital",
                    "total_amount": 0.0,
                    "date_of_service": "2024-01-01",
                    "patient_name": "Unknown Patient",
                    "diagnosis": "Unknown Diagnosis",
                    "admission_date": "2024-01-01",
                    "discharge_date": "2024-01-01",
                }

            # Ensure we have default values for validation and decision
            if not validation_result:
                validation_result = {"missing_documents": [], "discrepancies": []}

            if not claim_decision:
                claim_decision = {"status": "rejected", "reason": "Decision processing failed"}

            # Structure the result for the router
            final_results.append({"extracted_fields": extracted_data, "validation_result": validation_result, "claim_decision": claim_decision})

    except Exception as e:
        logger.error(f"Error in multi-agent pipeline: {e}")
        raise

    return final_results
