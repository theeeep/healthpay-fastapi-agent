import json
import uuid
from typing import Dict, List

import dotenv
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.core.logger import logger
from app.core.utils import clean_json_response
from app.module.process_claim.prompts.prompt_manager import prompt_manager

dotenv.load_dotenv()

# Configure session service
session_service = InMemorySessionService()


# Step 3: Enhanced Validation Agent (Multi-Agent Orchestration)
validation_agent = LlmAgent(
    name="EnhancedDataValidator",
    model="gemini-2.5-flash",
    description="Enhanced validation with multi-agent orchestration for medical claims",
    instruction=prompt_manager.get_prompt("validate_claim_package"),
    output_key="validation_result",
)

# Step 4: Enhanced Decision Agent (Multi-Agent Orchestration)
decision_agent = LlmAgent(
    name="EnhancedClaimDecisionMaker",
    model="gemini-2.5-flash",
    description="Enhanced claim decision making with multi-agent orchestration",
    instruction=prompt_manager.get_prompt("make_claim_decision"),
    output_key="claim_decision",
)

# Create the enhanced sequential pipeline (only validation and decision)
enhanced_processing_pipeline = SequentialAgent(name="EnhancedProcessingPipeline", sub_agents=[validation_agent, decision_agent])


async def run_adk_claim_pipeline(genai_extracted_documents: List[Dict], user_id: str = None) -> List[Dict]:
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
                        logger.error(f"Failed to parse validation_result: {e}")
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

                        # Validate that claim_decision is a dict, not a list
                        if isinstance(claim_decision, list):
                            logger.warning(f"Claim decision is a list: {claim_decision}")
                            if claim_decision:
                                claim_decision = claim_decision[0]  # Take first item
                            else:
                                claim_decision = {"status": "rejected", "reason": "No valid decision returned"}

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse claim_decision: {e}")
                        claim_decision = {"status": "rejected", "reason": "Failed to parse decision"}

        # Create final result combining validation and decision
        if validation_result or claim_decision:
            final_result = {
                "extracted_fields": None,  # ADK doesn't extract, it validates
                "validation_result": validation_result,
                "claim_decision": claim_decision,
            }
            final_results.append(final_result)
            logger.info(
                f"Added ADK result with validation: {bool(validation_result)}, decision: {claim_decision.get('status', 'unknown') if isinstance(claim_decision, dict) else 'unknown'}"
            )

        logger.info(f"Enhanced ADK Pipeline completed with {len(final_results)} results")

    except Exception as e:
        logger.error(f"Error running enhanced ADK claim processing pipeline: {e}")
        raise

    return final_results
