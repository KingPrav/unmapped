"""
PIPELINE 1 - Occupation Classifier
Data source: ISCO-08 taxonomy (seed data + real data from data/processed/)
Takes user's natural language description -> returns ISCO-08 occupation code
"""

import json
from pathlib import Path

from app.services.llm import chat_json, has_llm_credentials

BASE_DIR = Path(__file__).parent.parent
SEED_PATH = BASE_DIR / "data" / "seed" / "seed_data.json"

# Load seed occupations
with open(SEED_PATH) as f:
    SEED_DATA = json.load(f)


def _build_occupation_list() -> str:
    """Build occupation context for the classifier prompt."""
    lines = []
    for code, occ in SEED_DATA["occupations"].items():
        keywords = ", ".join(occ["common_names"])
        lines.append(f"- ISCO {code}: {occ['title']} (also known as: {keywords})")
    return "\n".join(lines)


OCCUPATION_LIST = _build_occupation_list()

CLASSIFY_PROMPT = """You are helping classify informal workers' occupations into standard ISCO-08 codes.

Available occupations:
{occupation_list}

User described their work as: "{user_description}"

Instructions:
1. Identify the best matching occupation from the list above
2. Return the ISCO code, title, and your confidence (0.0-1.0)
3. If unsure, pick the closest match
4. Return ONLY valid JSON, no explanation

Return this exact JSON format:
{{
  "isco_code": "XXXX",
  "title": "occupation title",
  "confidence": 0.0,
  "matched_on": "brief reason why this matched"
}}"""


def classify_occupation(user_description: str) -> dict:
    """
    Pipeline 1: Maps free-text occupation description to ISCO-08 code.

    Args:
        user_description: What the user says they do (e.g. "I fix phones")

    Returns:
        dict with isco_code, title, confidence, matched_on, and full occupation data
    """
    if not has_llm_credentials():
        raise ValueError("No LLM API key found. Set GROQ_API_KEY or OPENAI_API_KEY in your .env file.")

    prompt = CLASSIFY_PROMPT.format(
        occupation_list=OCCUPATION_LIST,
        user_description=user_description,
    )

    result = chat_json(
        system="You are a helpful assistant that always responds with valid JSON only.",
        user=prompt,
        max_tokens=256,
        temperature=0.1,
        preferred_openai_model="gpt-4o-mini",
    )
    isco_code = result["isco_code"]

    # Attach full occupation data from seed
    result["occupation_data"] = SEED_DATA["occupations"].get(isco_code)
    return result
