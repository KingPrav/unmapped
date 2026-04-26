"""
PIPELINE 6 - Localiser & Profile Generator
Data sources: ESCO taxonomy (language labels) + Country config
Generates a structured, portable Skills Card - something Amara can own, show,
and carry across borders.
"""

import uuid
from datetime import datetime

from app.services.llm import chat_json

EDUCATION_LABELS = {
    "none": "No formal education",
    "primary": "Primary school",
    "lower_secondary": "Lower secondary (JSS/JHS)",
    "upper_secondary": "Upper secondary (SSS/SHS)",
    "vocational": "Vocational / Technical",
    "tertiary": "Tertiary / University",
}


def _education_label(education_level: str, country_config: dict) -> str:
    for item in country_config.get("education_levels", []):
        if item.get("value") == education_level:
            return item.get("label", education_level)
    return EDUCATION_LABELS.get(education_level, education_level)

TIER_LABELS = {1: "Entry", 2: "Functional", 3: "Mastery"}
TIER_COLORS = {1: "#f97316", 2: "#6366f1", 3: "#22c55e"}

SUMMARY_PROMPT = """You are writing the human-readable summary section of a portable skills card for an informal economy worker.
This card is the person's proof of competency - they will show it to employers, training programs, and community organisations.

Worker:
- Occupation: {occupation_title}
- Education: {education_label}
- Experience: {experience_years} years
- Other skills: {other_skills}
- Location: {location_context}

Assessment results:
{dimension_results}

Write TWO things:

1. SUMMARY (2 sentences max): A plain, human description of who this person is and what they can demonstrably do. Write it as if you're introducing them to an employer. Use "This person" not "Amara". Do not mention scores or tiers.

2. EMPLOYER SIGNAL (1 sentence): What kind of role or responsibility is this person ready for RIGHT NOW, honestly and specifically. No aspirational language - only what the evidence supports.

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
    spoken_language: str = "",
    isco_code: str = "0000",
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
    education_label = _education_label(education_level, country_config)
    formatted_results = _format_dimension_results(dimension_results)

    prompt = SUMMARY_PROMPT.format(
        occupation_title=occupation_title,
        education_label=education_label,
        experience_years=experience_years,
        other_skills=other_skills if other_skills else "Not specified",
        location_context=country_config["location_context"],
        dimension_results=formatted_results,
    )

    summary_data = chat_json(
        system="You write concise, honest, human-readable skill summaries. Always return valid JSON.",
        user=prompt,
        max_tokens=300,
        temperature=0.4,
        preferred_openai_model="gpt-4o",
    )

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
            "feedback": result["feedback"],
        })
        total_score += score

        if score >= 65:
            strengths.append(result["dimension_label"])
        else:
            growth_areas.append(result["dimension_label"])

    avg_score = int(total_score / len(dimension_results)) if dimension_results else 0
    profile_id = _generate_profile_id(isco_code, country_config.get("country_code", "XX"))

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
            "location": country_config["location_context"],
        },
        "assessment": {
            "overall_score": avg_score,
            "dimensions": dimension_summary,
            "strengths": strengths,
            "growth_areas": growth_areas,
        },
        "readable": {
            "summary": summary_data.get("summary", ""),
            "employer_signal": summary_data.get("employer_signal", ""),
        },
        "portability": {
            "taxonomy": "ISCO-08 / O*NET / ESCO v1.2.1",
            "data_sources": country_config.get("data_sources", {}),
            "transferable_to": [
                "employer verification",
                "training enrollment",
                "financial services",
                "cross-border employment",
            ],
        },
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
        "spoken_language": spoken_language,
        "skills_card": skills_card,
        "data_sources": country_config.get("data_sources", {}),
        "opportunity_types": country_config.get("opportunity_types", []),
    }
