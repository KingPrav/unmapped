"""
PIPELINE 1 — Occupation Classifier
Data source: ISCO-08 taxonomy (seed data + real data from data/processed/)
Takes user's natural language description → returns ISCO-08 occupation code
"""

import json
import os
from openai import OpenAI
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SEED_PATH = BASE_DIR / "data" / "seed" / "seed_data.json"

# Load seed occupations
with open(SEED_PATH) as f:
    SEED_DATA = json.load(f)

# Build occupation context for LLM prompt
def _build_occupation_list() -> str:
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


def _parse_json_response(text: str) -> dict:
    """Safely parse JSON from OpenAI — strips markdown code blocks if present."""
    text = text.strip()
    # Strip markdown code fences OpenAI sometimes wraps around JSON
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    return json.loads(text.strip())


def classify_occupation(user_description: str) -> dict:
    """
    Pipeline 1: Maps free-text occupation description to ISCO-08 code.

    Args:
        user_description: What the user says they do (e.g. "I fix phones")

    Returns:
        dict with isco_code, title, confidence, matched_on, and full occupation data
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment. Check your .env file.")

    client = OpenAI(api_key=api_key)

    prompt = CLASSIFY_PROMPT.format(
        occupation_list=OCCUPATION_LIST,
        user_description=user_description
    )

    message = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=256,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant that always responds with valid JSON only."},
            {"role": "user", "content": prompt}
        ]
    )

    result = _parse_json_response(message.choices[0].message.content)
    isco_code = result["isco_code"]

    # Attach full occupation data from seed
    if isco_code in SEED_DATA["occupations"]:
        result["occupation_data"] = SEED_DATA["occupations"][isco_code]
    else:
        result["occupation_data"] = None

    return result
