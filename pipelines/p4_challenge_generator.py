"""
PIPELINE 4 — Challenge Generator
Data source: O*NET DWAs (via dimension plan) + Country config
Takes skill dimension + DWA + country context → generates scenario-based challenge question
Uses Claude API for generation
"""

import json
import os
from openai import OpenAI
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def _parse_json_response(text: str) -> dict:
    """Safely parse JSON — strips markdown code blocks if OpenAI wraps them."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    return json.loads(text.strip())

CHALLENGE_PROMPT = """You are generating a practical skills assessment question for an informal economy worker.

Worker occupation: {occupation_title}
Skill dimension being assessed: {dimension_label}
What this dimension measures: {dimension_description}
Specific work task this question is based on: "{primary_dwa}"

Local context:
- Location: {location_context}
- Tools available: {local_tools}
- Currency: {currency}
- Difficulty level: Tier {tier} ({tier_description})

Generate ONE scenario-based question that:
1. Describes a realistic work situation the person would actually encounter
2. Requires genuine domain knowledge — someone without experience cannot guess well
3. Is answerable in 3-6 sentences without internet access or formal tools
4. Reflects the actual working conditions and tools available in {location_context}
5. Does NOT reference formal certifications, diplomas, or institutional systems
6. Uses plain, clear language appropriate for the context

Tier guidance:
- Tier 1 (Entry): Basic competency — can they handle a routine situation correctly?
- Tier 2 (Functional): Intermediate — can they diagnose, decide, and act with partial information?
- Tier 3 (Mastery): Advanced — can they handle a complex or ambiguous situation and explain their reasoning?

Return ONLY the question text. No preamble, no labels, no explanation."""

EVAL_PROMPT = """You are evaluating a skills assessment answer from an informal economy worker.

Occupation: {occupation_title}
Skill dimension: {dimension_label}
Question asked: {question}
Tier of question: {tier}
Worker's answer: "{user_answer}"

Evaluate based on these criteria:
1. Demonstrates genuine domain knowledge (not just common sense)
2. Shows systematic thinking (considers cause, not just symptom)
3. Is practically sound (would actually work in the real context)

Return ONLY this JSON format:
{{
  "score": <integer 0-100>,
  "tier_achieved": <1, 2, or 3>,
  "passed": <true or false>,
  "feedback": "<one constructive sentence — what they showed and what could be stronger>",
  "next_tier": <1, 2, or 3 — what tier to challenge them with next>
}}"""


TIER_DESCRIPTIONS = {
    1: "Entry level — routine situation, standard response expected",
    2: "Functional — partial information, requires diagnosis and judgment",
    3: "Mastery — complex or ambiguous, requires reasoning and explanation"
}


def generate_challenge(
    occupation_title: str,
    dimension: dict,
    country_config: dict,
    tier: int = 1
) -> dict:
    """
    Pipeline 4: Generates a scenario-based challenge question.

    Args:
        occupation_title: e.g. "Electronics mechanics and servicers"
        dimension: dimension plan entry with label, description, primary_dwa
        country_config: loaded country YAML config
        tier: 1, 2, or 3 (adaptive difficulty)

    Returns:
        dict with challenge question and metadata
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = CHALLENGE_PROMPT.format(
        occupation_title=occupation_title,
        dimension_label=dimension["label"],
        dimension_description=dimension["description"],
        primary_dwa=dimension["primary_dwa"],
        location_context=country_config["location_context"],
        local_tools=country_config["local_tools"],
        currency=country_config["currency"],
        tier=tier,
        tier_description=TIER_DESCRIPTIONS[tier]
    )

    message = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    question_text = message.choices[0].message.content.strip()

    return {
        "question": question_text,
        "dimension_id": dimension["dimension_id"],
        "dimension_label": dimension["label"],
        "employer_signal": dimension["employer_signal"],
        "tier": tier,
        "occupation_title": occupation_title,
        "region": country_config["region_id"]
    }


def evaluate_answer(
    challenge: dict,
    user_answer: str,
    occupation_title: str
) -> dict:
    """
    Pipeline 4b: Evaluates a user's answer against the challenge.

    Args:
        challenge: The challenge dict from generate_challenge()
        user_answer: The worker's text answer
        occupation_title: For context in evaluation

    Returns:
        dict with score, tier_achieved, passed, feedback, next_tier
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = EVAL_PROMPT.format(
        occupation_title=occupation_title,
        dimension_label=challenge["dimension_label"],
        question=challenge["question"],
        tier=challenge["tier"],
        user_answer=user_answer
    )

    message = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=256,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are an evaluator that always responds with valid JSON only."},
            {"role": "user", "content": prompt}
        ]
    )

    result = _parse_json_response(message.choices[0].message.content)
    result["dimension_id"] = challenge["dimension_id"]
    result["dimension_label"] = challenge["dimension_label"]
    result["employer_signal"] = challenge["employer_signal"]

    return result
