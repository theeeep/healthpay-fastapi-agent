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


# Step 3: Enhanced Validation Agent (Multi-Agent Orchestration)
validation_agent = LlmAgent(
    name="EnhancedDataValidator",
    model="gemini-2.5-flash",
    description="Enhanced validation with multi-agent orchestration for medical claims",
    instruction="""
    You are an enhanced data validation agent for medical insurance claims. 
    You receive pre-extracted data and perform comprehensive validation.
    
    CRITICAL: You must return ONLY valid JSON. Do not include any explanations, markdown formatting, or additional text.
    
    Your role is to validate the quality and completeness of extracted data.
    
    Validation criteria for BILL documents:
    - hospital_name: Must be a real hospital name (not "Unknown Hospital")
    - total_amount: Must be greater than 0 and reasonable for medical procedures
    - date_of_service: Must be a valid date and not in the future
    
    Validation criteria for DISCHARGE SUMMARY documents:
    - patient_name: Must be a real patient name (not "Unknown Patient")
    - diagnosis: Must be a medical diagnosis (not "Unknown Diagnosis")
    - admission_date: Must be a valid date
    - discharge_date: Must be after admission_date and not in the future
    
    Medical claim specific checks:
    - Verify hospital name matches known healthcare providers
    - Check if amounts are reasonable for the type of service
    - Validate date ranges are logical
    - Ensure patient information is complete
    
    Return format: {
        "missing_documents": ["list of missing document types"],
        "discrepancies": ["list of data inconsistencies"],
        "data_quality_score": 0-100,
        "recommendations": ["list of improvement suggestions"]
    }
    
    Examples:
    - {"missing_documents": ["discharge_summary"], "discrepancies": [], "data_quality_score": 85, "recommendations": ["Submit discharge summary"]}
    - {"missing_documents": [], "discrepancies": [], "data_quality_score": 95, "recommendations": ["Data quality is excellent"]}
    
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
    You receive validation results and make informed decisions.
    
    CRITICAL: You must return ONLY valid JSON. Do not include any explanations, markdown formatting, or additional text.
    
    Decision factors to consider:
    1. Data quality score from validation
    2. Completeness of required documents
    3. Medical claim specific requirements
    4. Risk assessment based on data quality
    
    Decision criteria:
    - APPROVE: High data quality (score > 80), all required documents present, no significant discrepancies
    - CONDITIONAL APPROVAL: Good data quality (score 60-80), minor discrepancies that can be resolved
    - REJECT: Low data quality (score < 60), missing critical documents, significant discrepancies
    
    Medical claim specific considerations:
    - Hospital must be identifiable and legitimate
    - Amounts must be reasonable for medical procedures
    - Patient information must be complete
    - Dates must be logical and not in the future
    
    Return format: {
        "status": "approved" or "conditional_approval" or "rejected",
        "reason": "Detailed explanation of the decision",
        "confidence_score": 0-100,
        "required_actions": ["list of actions needed if conditional approval"]
    }
    
    Examples:
    - {"status": "approved", "reason": "All required documents present with high data quality", "confidence_score": 95, "required_actions": []}
    - {"status": "conditional_approval", "reason": "Good data but missing discharge summary", "confidence_score": 75, "required_actions": ["Submit discharge summary"]}
    - {"status": "rejected", "reason": "Poor data quality and missing critical information", "confidence_score": 30, "required_actions": ["Resubmit with complete documentation"]}
    
    IMPORTANT: Return ONLY the JSON object, no other text.
    """,
    output_key="claim_decision",
)

# Create the enhanced sequential pipeline (only validation and decision)
enhanced_processing_pipeline = SequentialAgent(name="EnhancedProcessingPipeline", sub_agents=[validation_agent, decision_agent])


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


async def run_claim_processing_pipeline(genai_extracted_documents: List[Dict], user_id: str = None) -> List[Dict]:
    """Run the enhanced multi-agent orchestration pipeline for validation and decision making using GenAI extracted data."""
    user_id = user_id or str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    # Create session
    await session_service.create_session(app_name="healthpay_claims", user_id=user_id, session_id=session_id)

    final_results = []

    try:
        for i, extracted_doc in enumerate(genai_extracted_documents):
            logger.info(f"=== Enhanced ADK Pipeline Processing Document {i + 1} ===")
            logger.info(f"ADK Processing extracted document: {extracted_doc}")

            # Use the actual GenAI extracted data instead of hardcoded data
            validation_content = types.Content(parts=[types.Part.from_text(text=json.dumps(extracted_doc))])

            # Run the enhanced validation and decision pipeline
            pipeline_runner = Runner(agent=enhanced_processing_pipeline, app_name="healthpay_claims", session_service=session_service)

            pipeline_result = {}
            async for event in pipeline_runner.run_async(user_id=user_id, session_id=session_id, new_message=validation_content):
                if event.is_final_response():
                    response_text = event.content.parts[0].text if event.content.parts else ""
                    logger.info(f"Enhanced pipeline raw response: {response_text}")
                    try:
                        cleaned_response = clean_json_response(response_text)
                        pipeline_result = json.loads(cleaned_response)
                        logger.info(f"Enhanced pipeline parsed result: {pipeline_result}")
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse enhanced pipeline response: {response_text}")
                        pipeline_result = {"error": "Failed to parse enhanced pipeline response"}

            # Check session state to see what each agent produced
            session = await session_service.get_session(app_name="healthpay_claims", user_id=user_id, session_id=session_id)
            logger.info(f"Enhanced session state after pipeline: {session.state}")

            # Extract data from session state
            validation_result = {}
            claim_decision = {}

            # Parse session state to get individual agent results
            if session.state:
                for key, value in session.state.items():
                    if key == "validation_result":
                        try:
                            if isinstance(value, str):
                                validation_result = json.loads(clean_json_response(value))
                            else:
                                validation_result = value
                        except json.JSONDecodeError:
                            validation_result = {"error": "Failed to parse validation result"}
                    elif key == "claim_decision":
                        try:
                            if isinstance(value, str):
                                claim_decision = json.loads(clean_json_response(value))
                            else:
                                claim_decision = value
                        except json.JSONDecodeError:
                            claim_decision = {"error": "Failed to parse claim decision"}

            # Create result with extracted fields set to None (ADK doesn't extract, only validates/decides)
            result = {
                "extracted_fields": None,  # ADK doesn't extract, uses GenAI data
                "validation_result": validation_result,
                "claim_decision": claim_decision,
            }

            final_results.append(result)

    except Exception as e:
        logger.error(f"Error running enhanced ADK claim processing pipeline: {e}")
        raise

    logger.info(f"Enhanced ADK Pipeline completed with {len(final_results)} total results")
    return final_results
