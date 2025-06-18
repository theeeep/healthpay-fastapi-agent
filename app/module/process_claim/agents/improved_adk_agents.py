import json
import uuid
from typing import Dict, List

import dotenv
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.core.logger import logger
from app.core.utils import clean_json_response, parse_json_safely

dotenv.load_dotenv()

# Configure session service
session_service = InMemorySessionService()


# Step 3: Enhanced Validation Agent (Multi-Agent Orchestration)
validation_agent = LlmAgent(
    name="EnhancedDataValidator",
    model="gemini-2.5-flash",
    description="Enhanced validation with multi-agent orchestration for medical claims",
    instruction="""
    You are an enhanced data validation agent for medical insurance claims. 
    You receive a complete claim package with multiple extracted documents and perform comprehensive validation.
    
    CRITICAL: You must return ONLY valid JSON. Do not include any explanations, markdown formatting, or additional text.
    
    Your role is to validate the quality and completeness of the entire claim package.
    
    Input format:
    {
        "extracted_documents": [
            {
                "type": "bill",
                "hospital_name": "Hospital Name",
                "total_amount": 12345.67,
                "date_of_service": "2025-02-11"
            },
            {
                "type": "discharge_summary",
                "patient_name": "Patient Name",
                "diagnosis": "Medical Diagnosis",
                "admission_date": "2025-02-07",
                "discharge_date": "2025-02-11"
            }
        ],
        "document_count": 2,
        "document_types": ["bill", "discharge_summary"]
    }
    
    Validation criteria for COMPLETE CLAIM PACKAGE:
    - Must have BOTH bill and discharge_summary documents
    - Bill document: hospital_name (not "Unknown Hospital"), total_amount > 0, valid date_of_service
    - Discharge summary: patient_name (not "Unknown Patient"), diagnosis (not "Unknown Diagnosis"), valid dates
    - Data consistency between documents (same patient, hospital, dates)
    
    Medical claim specific checks:
    - Verify hospital name matches known healthcare providers
    - Check if amounts are reasonable for the type of service
    - Validate date ranges are logical
    - Ensure patient information is complete and consistent
    
    Return format: {
        "missing_documents": ["list of missing document types"],
        "discrepancies": ["list of data inconsistencies"],
        "data_quality_score": 0-100,
        "recommendations": ["list of improvement suggestions"]
    }
    
    Examples:
    - {"missing_documents": [], "discrepancies": [], "data_quality_score": 95, "recommendations": ["Data quality is excellent"]}
    - {"missing_documents": ["discharge_summary"], "discrepancies": [], "data_quality_score": 85, "recommendations": ["Submit discharge summary"]}
    - {"missing_documents": ["bill"], "discrepancies": [], "data_quality_score": 75, "recommendations": ["Submit bill document"]}
    
    IMPORTANT: Return ONLY the JSON object, no other text.
    """,
    output_key="validation_result",
)

# Step 4: Enhanced Decision Agent (Multi-Agent Orchestration)
decision_agent = LlmAgent(
    name="EnhancedClaimDecisionMaker",
    model="gemini-2.5-flash",
    description="Enhanced claim decision making with multi-agent orchestration",
    instruction="""
    You are an enhanced claim decision agent for medical insurance claims.
    You receive validation results for a complete claim package and make informed decisions.
    
    CRITICAL: You must return ONLY valid JSON. Do not include any explanations, markdown formatting, or additional text.
    
    Decision factors to consider:
    1. Data quality score from validation
    2. Completeness of required documents (bill + discharge_summary)
    3. Medical claim specific requirements
    4. Risk assessment based on data quality
    
    Decision criteria:
    - APPROVE: High data quality (score > 80), both bill and discharge_summary present, no significant discrepancies
    - CONDITIONAL APPROVAL: Good data quality (score 60-80), minor discrepancies that can be resolved
    - REJECT: Low data quality (score < 60), missing critical documents (bill or discharge_summary), significant discrepancies
    
    Medical claim specific considerations:
    - Both bill and discharge_summary must be present
    - Hospital must be identifiable and legitimate
    - Amounts must be reasonable for medical procedures
    - Patient information must be complete and consistent
    - Dates must be logical and not in the future
    
    Return format: {
        "status": "approved" or "conditional_approval" or "rejected",
        "reason": "Detailed explanation of the decision",
        "confidence_score": 0-100,
        "required_actions": ["list of actions needed if conditional approval"]
    }
    
    IMPORTANT: For APPROVED claims, use exactly this reason: "All required documents present and data is consistent"
    
    Examples:
    - {"status": "approved", "reason": "All required documents present and data is consistent", 
       "confidence_score": 95, "required_actions": []}
    - {"status": "conditional_approval", "reason": "Good data but minor discrepancies in dates", 
       "confidence_score": 75, "required_actions": ["Verify admission/discharge dates"]}
    - {"status": "rejected", "reason": "Missing discharge summary document", 
       "confidence_score": 30, "required_actions": ["Submit complete discharge summary"]}
    
    IMPORTANT: Return ONLY the JSON object, no other text.
    """,
    output_key="claim_decision",
)

# Create the enhanced sequential pipeline (only validation and decision)
enhanced_processing_pipeline = SequentialAgent(name="EnhancedProcessingPipeline", sub_agents=[validation_agent, decision_agent])


async def run_claim_processing_pipeline(genai_extracted_documents: List[Dict], user_id: str = None) -> List[Dict]:
    """Run the enhanced multi-agent orchestration pipeline for validation and decision making using GenAI extracted data."""
    user_id = user_id or str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    # Create session
    await session_service.create_session(app_name="healthpay_claims", user_id=user_id, session_id=session_id)

    final_results = []

    try:
        logger.info("=== Enhanced ADK Pipeline Processing Complete Claim Package ===")
        logger.info(f"ADK Processing {len(genai_extracted_documents)} extracted documents")

        # Process ALL extracted documents together as a complete claim package
        complete_claim_data = {
            "extracted_documents": genai_extracted_documents,
            "document_count": len(genai_extracted_documents),
            "document_types": [doc.get("type") for doc in genai_extracted_documents],
        }

        # Create content with the complete claim data for validation
        validation_content = types.Content(parts=[types.Part.from_text(text=json.dumps(complete_claim_data))])

        # Run the enhanced validation and decision pipeline
        pipeline_runner = Runner(agent=enhanced_processing_pipeline, app_name="healthpay_claims", session_service=session_service)

        pipeline_result = {}
        async for event in pipeline_runner.run_async(user_id=user_id, session_id=session_id, new_message=validation_content):
            if event.is_final_response():
                response_text = event.content.parts[0].text if event.content.parts else ""
                try:
                    cleaned_response = clean_json_response(response_text)
                    parsed_result = json.loads(cleaned_response)
                    # Only set pipeline_result if it's a dictionary, not a list
                    if isinstance(parsed_result, dict):
                        pipeline_result = parsed_result
                    else:
                        logger.warning(f"Pipeline returned non-dict result: {type(parsed_result)}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse enhanced pipeline response: {response_text}")
                    logger.error(f"JSON decode error: {e}")
                    pipeline_result = {"error": "Failed to parse enhanced pipeline response"}

        # Check session state to see what each agent produced
        session = await session_service.get_session(app_name="healthpay_claims", user_id=user_id, session_id=session_id)
        logger.info(f"Enhanced session state after pipeline: {session.state}")

        # Extract data from session state
        validation_result = {}
        claim_decision = {}

        # Parse session state to get individual agent results
        if session.state:
            logger.info(f"Session state keys: {list(session.state.keys())}")
            for key, value in session.state.items():
                logger.info(f"Processing session key: {key}, value type: {type(value)}")
                if key == "validation_result":
                    try:
                        if isinstance(value, str):
                            logger.info(f"Validation result string: {value}")
                            validation_result = json.loads(clean_json_response(value))
                        else:
                            validation_result = value
                        logger.info(f"Parsed validation_result: {validation_result}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse validation result: {e}")
                        validation_result = {"error": "Failed to parse validation result"}
                elif key == "claim_decision":
                    try:
                        if isinstance(value, str):
                            logger.info(f"Claim decision string: {value}")
                            # Use the original clean_json_response function
                            cleaned_value = clean_json_response(value)
                            logger.info(f"Cleaned claim decision: {cleaned_value}")
                            claim_decision = json.loads(cleaned_value)
                        else:
                            claim_decision = value
                        logger.info(f"Parsed claim_decision: {claim_decision}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse claim decision: {e}")
                        logger.error(f"Raw claim decision value: {value}")
                        claim_decision = {"error": "Failed to parse claim decision"}
        else:
            logger.warning("Session state is empty - no agent results found")

        # Create result with extracted fields set to None (ADK doesn't extract, uses GenAI data)
        result = {
            "extracted_fields": None,  # ADK doesn't extract, uses GenAI data
            "validation_result": validation_result,
            "claim_decision": claim_decision,
        }

        logger.info(f"ADK final result: {result}")
        final_results.append(result)

    except Exception as e:
        logger.error(f"Error running enhanced ADK claim processing pipeline: {e}")
        raise

    logger.info(f"Enhanced ADK Pipeline completed with {len(final_results)} total results")
    return final_results
