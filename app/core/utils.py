import json
import re
from typing import Any


def clean_json_response(response_text: str) -> str:
    """Clean JSON response that might be wrapped in markdown code blocks or have explanatory text."""
    # Remove markdown code block formatting
    response_text = re.sub(r"^```json\s*", "", response_text)
    response_text = re.sub(r"\s*```$", "", response_text)
    response_text = response_text.strip()

    # PRIORITY: Look for object starting with { first (most common case)
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
                    return response_text[start_idx : i + 1]

    # FALLBACK: Look for array starting with [ (less common)
    array_match = re.search(r"\[[\s\S]*\]", response_text, re.DOTALL)
    if array_match:
        return array_match.group(0)

    # If no JSON found, return the original text stripped
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
