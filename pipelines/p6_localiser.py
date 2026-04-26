"""
PIPELINE 6 — Localiser & Profile Generator
Data sources: ESCO taxonomy (language labels) + Country config
Generates a structured, portable Skills Card — something Amara can own, show, and carry across borders.
"""

import json
import os
import sys
import uuid
from datetime import datetime
from openai import OpenAI
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))
from pipelines.scoring import compute_weighted_score

EDUCATION_LABELS = {
    "none": "No formal education",
    "primary": "Primary school",
    "lower_secondary": "Lower secondary (JSS/JHS)",
    "upper_secondary": "Upper secondary (SSS/SHS)",
    "vocational": "Vocational / Technical",
    "tertiary": "Tertiary / University"
}

TIER_LABELS = {1: "Entry", 2: "Functional", 3: "Mastery"}
TIER_COLORS = {1: "#f97316", 2: "#6366f1", 3: "#22c55e"}

SUMMARY_PROMPT = """You are writing the human-readable summary section of a portable skills card for an informal economy worker.
This card is the person's proof of competency — they will show it to employers, training programs, and community organisations.

Worker:
- Occupation: {occupation_title}
- Education: {education_label}
- Experience: {experience_years} years
- Other skills: {other_skills}
- Location: {location_context}

Assessment results:
{dimension_results}

LANGUAGE REQUIREMENT:
- Write BOTH the summary and employer_signal in {language_name}.
- Use {language_register}.
- If {language_name} is English, write naturally in English.
- Do not mix languages.

Write TWO things:

1. SUMMARY (2 sentences max): A plain, human description of who this person is and what they can demonstrably do. Write it as if you're introducing them to an employer. Use "This person" not "Amara". Do not mention scores or tiers.

2. EMPLOYER SIGNAL (1 sentence): What kind of role or responsibility is this person ready for RIGHT NOW, honestly and specifically. No aspirational language — only what the evidence supports.

Return as JSON:
{{
  "summary": "...",
  "employer_signal": "..."
}}"""


def _format_dimension_results(dimension_results: dict) -> str:
    lines = []
    for dim_id, result in dimension_results.items():
        tier_label = TIER_LABELS.get(result["tier_achieved"], "Entry")
        lines.append(
            f"- {result['dimension_label']}: {tier_label}\n"
            f"  Score: {result['score']}/100 | Feedback: {result['feedback']}"
        )
    return "\n".join(lines)


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    return json.loads(text.strip())


def _generate_profile_id(isco_code: str, country_code: str) -> str:
    """Generate a unique, portable profile ID."""
    short_id = str(uuid.uuid4())[:8].upper()
    year = datetime.now().year
    return f"UNM-{country_code}-{isco_code}-{year}-{short_id}"


def generate_profile(
    occupation_title: str,
    dimension_results: dict,
    country_config: dict,
    education_level: str = "upper_secondary",
    experience_years: int = 0,
    other_skills: str = "",
    isco_code: str = "0000"
) -> dict:
    """
    Pipeline 6: Generates a structured, portable Skills Card.

    Returns a full profile dict including:
    - Structured Skills Card (portable, human-readable)
    - AI-generated summary and employer signal
    - Dimension scores and tiers
    - Data source citations
    - Unique profile ID
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    education_label = EDUCATION_LABELS.get(education_level, education_level)
    formatted_results = _format_dimension_results(dimension_results)

    # Language settings from config — fall back to English
    language_name     = country_config.get("language_name", "English")
    language_register = country_config.get("language_register", "Standard English")

    # Generate human-readable summary + employer signal (in worker's language)
    prompt = SUMMARY_PROMPT.format(
        occupation_title=occupation_title,
        education_label=education_label,
        experience_years=experience_years,
        other_skills=other_skills if other_skills else "Not specified",
        location_context=country_config["location_context"],
        dimension_results=formatted_results,
        language_name=language_name,
        language_register=language_register,
    )

    system_prompt = (
        "You write concise, honest, human-readable skill summaries. Always return valid JSON. "
        f"The summary and employer_signal MUST be written entirely in {language_name}."
    )

    message = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=300,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    summary_data = _parse_json(message.choices[0].message.content)

    # Build structured dimension summary
    dimension_summary = []
    total_score = 0
    strengths = []
    growth_areas = []

    for dim_id, result in dimension_results.items():
        tier = result["tier_achieved"]
        score = result["score"]
        tier_label = TIER_LABELS.get(tier, "Entry")

        dimension_summary.append({
            "dimension_id": dim_id,
            "label": result["dimension_label"],
            "score": score,
            "tier": tier,
            "tier_label": tier_label,
            "tier_color": TIER_COLORS.get(tier, "#6366f1"),
            "passed": result["passed"],
            "employer_signal": result["employer_signal"],
            "feedback": result["feedback"]
        })
        total_score += score

        if score >= 65:
            strengths.append(result["dimension_label"])
        else:
            growth_areas.append(result["dimension_label"])

    # ISCO-aware weighted score — dimensions are weighted by occupational relevance,
    # not averaged equally. A phone repair tech's fault_diagnosis counts 40%;
    # a market seller's communication counts 40%.
    avg_score = compute_weighted_score(dimension_results, isco_code)
    profile_id = _generate_profile_id(isco_code, country_config.get("country_code", "XX"))

    # The structured Skills Card — portable across borders
    skills_card = {
        "profile_id": profile_id,
        "issued_date": datetime.now().strftime("%B %Y"),
        "issuing_system": "UNMAPPED Skills Infrastructure v1.0",
        "worker": {
            "occupation": occupation_title,
            "isco_code": isco_code,
            "education": education_label,
            "experience_years": experience_years,
            "other_skills": other_skills,
            "region": country_config["region_name"],
            "location": country_config["location_context"]
        },
        "assessment": {
            "overall_score": avg_score,
            "dimensions": dimension_summary,
            "strengths": strengths,
            "growth_areas": growth_areas
        },
        "readable": {
            "summary": summary_data.get("summary", ""),
            "employer_signal": summary_data.get("employer_signal", "")
        },
        "portability": {
            "taxonomy": "ISCO-08 / O*NET / ESCO v1.2.1",
            "data_sources": country_config.get("data_sources", {}),
            "transferable_to": ["employer verification", "training enrollment", "financial services", "cross-border employment"]
        }
    }

    return {
        "occupation_title": occupation_title,
        "region": country_config["region_name"],
        "location": country_config["location_context"],
        "overall_score": avg_score,
        "profile_id": profile_id,
        "summary": summary_data.get("summary", ""),
        "employer_signal": summary_data.get("employer_signal", ""),
        "dimension_summary": dimension_summary,
        "strengths": strengths,
        "growth_areas": growth_areas,
        "education": education_label,
        "experience_years": experience_years,
        "other_skills": other_skills,
        "skills_card": skills_card,
        "language": country_config.get("language", "en"),
        "language_name": language_name,
        "data_sources": country_config.get("data_sources", {}),
        "opportunity_types": country_config.get("opportunity_types", [])
    }
