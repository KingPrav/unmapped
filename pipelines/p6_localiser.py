"""
PIPELINE 6 — Localiser & Profile Generator
Data sources: ESCO taxonomy (language labels) + Country config
Adapts all output to local context and generates the final employer-readable skill profile.
"""

import json
import os
from openai import OpenAI
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

PROFILE_PROMPT = """You are generating a professional, employer-readable skill profile for an informal economy worker.
This profile is the core output of the UNMAPPED system — it makes an invisible worker visible to the formal economy.

Worker details:
- Occupation: {occupation_title}
- Location: {location_context}
- Region: {region_name}

Assessment results:
{dimension_results}

Generate a structured skill profile that:
1. Opens with a 1-sentence summary of who this person is and what they can do
2. Lists each assessed dimension with their tier level (Entry / Functional / Mastery)
3. Highlights 2-3 key strengths with specific evidence from their scores
4. Notes 1-2 growth areas honestly but constructively
5. Closes with an "Employer Signal" — a plain sentence about what kind of work they're ready for

Tone: Professional but human. This profile represents a real person.
Length: 200-250 words maximum.
Do NOT mention the assessment system, scores, or tiers in the employer signal section — just what they can do."""

TIER_LABELS = {
    1: "Entry",
    2: "Functional",
    3: "Mastery"
}


def _format_dimension_results(dimension_results: dict) -> str:
    """Format dimension results for the profile prompt."""
    lines = []
    for dim_id, result in dimension_results.items():
        tier_label = TIER_LABELS.get(result["tier_achieved"], "Entry")
        status = "✓ Passed" if result["passed"] else "○ In progress"
        lines.append(
            f"- {result['dimension_label']}: {tier_label} ({status})\n"
            f"  Score: {result['score']}/100\n"
            f"  Feedback: {result['feedback']}"
        )
    return "\n".join(lines)


def generate_profile(
    occupation_title: str,
    dimension_results: dict,
    country_config: dict
) -> dict:
    """
    Pipeline 6: Generates localised employer-readable skill profile.

    Args:
        occupation_title: Worker's occupation
        dimension_results: Dict of dimension assessments from Pipeline 5
        country_config: Loaded country YAML config

    Returns:
        dict with profile text, structured scores, and metadata
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    formatted_results = _format_dimension_results(dimension_results)

    prompt = PROFILE_PROMPT.format(
        occupation_title=occupation_title,
        location_context=country_config["location_context"],
        region_name=country_config["region_name"],
        dimension_results=formatted_results
    )

    message = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    profile_text = message.choices[0].message.content.strip()

    # Build structured score summary
    dimension_summary = []
    total_score = 0
    for dim_id, result in dimension_results.items():
        dimension_summary.append({
            "dimension_id": dim_id,
            "label": result["dimension_label"],
            "score": result["score"],
            "tier": result["tier_achieved"],
            "tier_label": TIER_LABELS.get(result["tier_achieved"], "Entry"),
            "passed": result["passed"],
            "employer_signal": result["employer_signal"]
        })
        total_score += result["score"]

    avg_score = int(total_score / len(dimension_results)) if dimension_results else 0

    return {
        "occupation_title": occupation_title,
        "region": country_config["region_name"],
        "location": country_config["location_context"],
        "overall_score": avg_score,
        "profile_text": profile_text,
        "dimension_summary": dimension_summary,
        "data_sources": country_config.get("data_sources", {}),
        "opportunity_types": country_config.get("opportunity_types", [])
    }
