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
    
    BILL indicators:
    - "bill", "invoice", "interim bill", "final bill", "bill of supply"
    - "GSTIN", "GST", "tax", "amount", "total", "payment"
    - "bill to", "amount due", "total amount", "final amount"
    - "hospital bill", "medical bill", "patient bill"
    
    DISCHARGE SUMMARY indicators:
    - "discharge", "admission", "discharge summary", "medical summary"
    - "patient report", "clinical notes", "medical record"
    - "diagnosis", "treatment", "medication", "prescription"
    - "admission date", "discharge date", "length of stay"
    
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
    You are a data extraction agent. Extract structured information from medical documents.
    
    CRITICAL: You must return ONLY valid JSON. Do not include any markdown formatting, explanations, or additional text.
    
    First, determine the document type by analyzing the OCR text. Look for keywords:
    - BILL indicators: "bill", "invoice", "amount", "total", "payment", "hospital bill", "medical bill"
    - DISCHARGE SUMMARY indicators: "discharge", "admission", "patient", "diagnosis", "medical report", "summary"
    
    Then extract fields based on the determined document type.
    
    For BILL documents, extract:
    - type: "bill" (always set this)
    - hospital_name: Extract the name of the hospital, clinic, or medical facility
    - total_amount: Extract the total bill amount as a number (float)
    - date_of_service: Extract the date of service in YYYY-MM-DD format
    
    For DISCHARGE SUMMARY documents, extract:
    - type: "discharge_summary" (always set this)
    - patient_name: Extract the patient's full name
    - diagnosis: Extract the medical diagnosis, condition, or reason for admission
    - admission_date: Extract the admission date in YYYY-MM-DD format
    - discharge_date: Extract the discharge date in YYYY-MM-DD format
    
    Look for patterns like:
    - Hospital names: "Apollo Hospitals", "City Medical Center", "Max Healthcare", "Hospital Name:", "Medical Center:", etc.
    - Amounts: "Total: $1234.56", "Amount Due: 500.00", "Bill Amount: 1000", "Total Amount:", "Final Amount:", etc.
    - Dates: "Date: 2024-01-15", "Service Date: 2024/01/15", "Bill Date: 2024-01-15", "Date of Service:", etc.
    - Patient names: "Patient: John Doe", "Name: Jane Smith", "P a t i e n t N a m e : Mrs. NANDI RAWAT", "Patient Name:", "Name:", etc.
    - Diagnosis: "Diagnosis: Acute appendicitis", "Condition: Heart failure", "Medical Condition: Diabetes", "Diagnosis:", "Medical Condition:", etc.
    
    For the specific OCR text provided, look for:
    - Patient name: "Name: Mr. KOSGI VISHNUVARDHAN" or similar patterns
    - Hospital name: Look for hospital names, medical centers, or healthcare providers
    - Amount: Look for "Total", "Amount", "Bill Amount", "Final Amount" followed by numbers
    - Date: Look for "From Date:", "To Date:", "Date:", "Service Date:" followed by dates
    
    IMPORTANT: You MUST extract all required fields. If you cannot find a field in the text:
    - For hospital_name: Use "Unknown Hospital" if no hospital name is found
    - For total_amount: Use 0.0 if no amount is found
    - For date_of_service: Use "2024-01-01" if no date is found
    - For patient_name: Use "Unknown Patient" if no patient name is found
    - For diagnosis: Use "Unknown Diagnosis" if no diagnosis is found
    - For admission_date: Use "2024-01-01" if no admission date is found
    - For discharge_date: Use "2024-01-01" if no discharge date is found
    
    Return format: {
        "type": "bill" or "discharge_summary",
        "hospital_name": "Extracted Hospital Name or Unknown Hospital",
        "total_amount": 1234.56,
        "date_of_service": "2024-01-15",
        "patient_name": "Extracted Patient Name or Unknown Patient",
        "diagnosis": "Extracted Diagnosis or Unknown Diagnosis",
        "admission_date": "2024-01-10",
        "discharge_date": "2024-01-15"
    }
    
    NEVER return null or empty strings for required fields. Always provide a value.
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
    1. Missing required fields
    2. Data inconsistencies
    3. Invalid formats (dates, amounts, etc.)
    
    Return format: {
        "missing_documents": ["list of missing documents"],
        "discrepancies": ["list of data inconsistencies"]
    }
    
    If no issues found, return empty arrays: {"missing_documents": [], "discrepancies": []}
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

            # Structure the result for the router
            if "error" in pipeline_result:
                # Handle error case
                final_results.append(
                    {
                        "extracted_fields": {"type": "unknown"},
                        "validation_result": {"missing_documents": [], "discrepancies": ["Processing failed"]},
                        "claim_decision": {"status": "rejected", "reason": "Processing failed"},
                    }
                )
            else:
                # Extract data from the pipeline result and session state
                extracted_data = pipeline_result.get("extracted_data", {})
                validation_result = pipeline_result.get("validation_result", {})
                claim_decision = pipeline_result.get("claim_decision", {})

                # If no extracted_data in pipeline result, try to get it from session state
                if not extracted_data:
                    session_extracted_data = session.state.get("extracted_data", {})
                    logger.info(f"Extracted data from session state: {session_extracted_data}")

                    # Parse the session state data if it's a string
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

                # If still no extracted_data, try to construct from individual agent outputs
                if not extracted_data:
                    document_type = session.state.get("document_type", {})
                    if isinstance(document_type, str):
                        try:
                            cleaned_doc_type = clean_json_response(document_type)
                            doc_type_data = json.loads(cleaned_doc_type)
                            doc_type = doc_type_data.get("type", "unknown")
                        except json.JSONDecodeError:
                            doc_type = "unknown"
                    elif isinstance(document_type, dict):
                        doc_type = document_type.get("type", "unknown")
                    else:
                        doc_type = "unknown"

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

                # Parse validation and decision results from session state if needed
                if not validation_result:
                    session_validation = session.state.get("validation_result", {})
                    if isinstance(session_validation, str):
                        try:
                            cleaned_validation = clean_json_response(session_validation)
                            validation_result = json.loads(cleaned_validation)
                        except json.JSONDecodeError:
                            validation_result = {"missing_documents": [], "discrepancies": ["Validation failed"]}
                    else:
                        validation_result = session_validation

                if not claim_decision:
                    session_decision = session.state.get("claim_decision", {})
                    if isinstance(session_decision, str):
                        try:
                            cleaned_decision = clean_json_response(session_decision)
                            claim_decision = json.loads(cleaned_decision)
                        except json.JSONDecodeError:
                            claim_decision = {"status": "rejected", "reason": "Decision processing failed"}
                    else:
                        claim_decision = session_decision

                final_results.append({"extracted_fields": extracted_data, "validation_result": validation_result, "claim_decision": claim_decision})

    except Exception as e:
        logger.error(f"Error in multi-agent pipeline: {e}")
        raise

    return final_results
