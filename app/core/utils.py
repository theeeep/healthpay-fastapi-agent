import json
import re
from typing import Any


def clean_json_response(response_text: str) -> str:
    """Clean JSON response that might be wrapped in markdown code blocks or have explanatory text."""
    logger = None
    try:
        from app.core.logger import logger
    except ImportError:
        pass

    if logger:
        logger.debug(f"Cleaning JSON response: {response_text[:200]}...")

    # Remove markdown code block formatting
    response_text = re.sub(r"^```json\s*", "", response_text)
    response_text = re.sub(r"\s*```$", "", response_text)
    response_text = response_text.strip()

    if logger:
        logger.debug(f"After markdown removal: {response_text[:200]}...")

    # Try to find JSON array first (for document extraction - priority)
    array_match = re.search(r"\[[\s\S]*\]", response_text, re.DOTALL)
    if array_match:
        try:
            # Validate it's actually JSON
            json.loads(array_match.group(0))
            # Only return non-empty arrays
            if array_match.group(0).strip() != "[]":
                if logger:
                    logger.debug(f"Found valid JSON array: {array_match.group(0)[:100]}...")
                return array_match.group(0)
            else:
                if logger:
                    logger.debug("Found empty array, skipping")
        except json.JSONDecodeError:
            if logger:
                logger.debug("JSON array validation failed")
            pass  # Continue to object search

    # Try to find JSON object (for ADK agents)
    start_idx = response_text.find("{")
    if start_idx != -1:
        # Count braces to find the matching closing brace
        brace_count = 0
        for i in range(start_idx, len(response_text)):
            if response_text[i] == "{":
                brace_count += 1
            elif response_text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    try:
                        # Validate it's actually JSON
                        json_obj = response_text[start_idx : i + 1]
                        json.loads(json_obj)
                        if logger:
                            logger.debug(f"Found valid JSON object: {json_obj[:100]}...")
                        return json_obj
                    except json.JSONDecodeError:
                        if logger:
                            logger.debug("JSON object validation failed")
                        pass  # Continue to other searches

    # If no valid JSON found, return the original text stripped
    if logger:
        logger.debug(f"No valid JSON found, returning stripped text: {response_text[:100]}...")
    return response_text.strip()


def parse_json_safely(text: str, default: Any = None) -> Any:
    """Safely parse JSON with error handling and logging."""
    try:
        cleaned_response = clean_json_response(text)
        return json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        from app.core.logger import logger

        logger.error(f"Failed to parse JSON response: {e}")
        logger.error(f"Raw text: {text}")
        return default
