"""
PIPELINE 7 — Opportunity Matcher
Data sources: opportunities_ghana.json / opportunities_kenya.json (ILOSTAT + WBES signals)
Matches a worker's skill profile to real local opportunities in 3 honest layers:
  - ready_now: skill tiers meet or exceed requirements
  - close_gap: 1-2 dimensions below requirement
  - training_pathway: structured training unlocks better outcomes
"""

import json
import os
from openai import OpenAI
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SEED_DIR = BASE_DIR / "data" / "seed"

TIER_LABELS = {1: "Entry", 2: "Functional", 3: "Mastery"}

MATCH_EXPLANATION_PROMPT = """You are writing a short, honest explanation of why a specific opportunity fits (or partially fits) a worker's skill profile.

Worker:
- Occupation: {occupation_title}
- Education: {education}
- Experience: {experience_years} years
- Region: {region}

Opportunity: {opportunity_title} ({opportunity_type})
Match layer: {match_layer}

Their skill assessment:
{dimension_summary}

Required for this opportunity:
{required_dimensions}

Gaps (if any):
{gaps}

Write 2-3 sentences that:
1. Acknowledge what they already have that makes this relevant
2. Are honest about any gap — state it plainly without sugarcoating
3. Give one specific, actionable next step they can take this week

Do NOT mention the assessment system by name. Write directly to the person. Be concise and human."""


def _load_opportunities(region_id: str) -> list:
    """Load opportunity seed data for the given region."""
    filename = f"opportunities_{region_id.split('_')[0]}.json"
    path = SEED_DIR / filename
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("opportunities", [])


def _load_meta(region_id: str) -> dict:
    filename = f"opportunities_{region_id.split('_')[0]}.json"
    path = SEED_DIR / filename
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("meta", {})


def _score_match(opportunity: dict, dimension_results: dict, isco_group: int) -> dict:
    """
    Score how well a worker's profile matches an opportunity.
    Returns match_score (0-100), gaps, and match_layer.
    """
    required = opportunity.get("required_dimensions", {})
    opp_isco_groups = opportunity.get("isco_groups", [])

    # ISCO group compatibility (soft match — adjacent groups allowed)
    isco_compatible = (
        isco_group in opp_isco_groups or
        any(abs(isco_group - g) <= 1 for g in opp_isco_groups)
    )

    if not required:
        # Training pathways have minimal requirements
        return {
            "match_score": 60,
            "gaps": [],
            "met": [],
            "isco_compatible": True,
            "match_layer": "training_pathway"
        }

    gaps = []
    met = []
    total_requirements = len(required)
    met_count = 0

    for dim_id, required_tier in required.items():
        actual = dimension_results.get(dim_id, {})
        actual_tier = actual.get("tier_achieved", 0)

        if actual_tier >= required_tier:
            met_count += 1
            met.append({
                "dimension": dim_id,
                "required": required_tier,
                "actual": actual_tier,
                "label": actual.get("dimension_label", dim_id)
            })
        else:
            gap_size = required_tier - actual_tier
            gaps.append({
                "dimension": dim_id,
                "required": required_tier,
                "required_label": TIER_LABELS.get(required_tier, "Entry"),
                "actual": actual_tier,
                "actual_label": TIER_LABELS.get(actual_tier, "Entry"),
                "label": actual.get("dimension_label", dim_id),
                "gap_size": gap_size
            })

    match_pct = met_count / total_requirements if total_requirements > 0 else 0
    match_score = int(match_pct * 100)

    # Determine match layer
    if match_score >= 100 and isco_compatible:
        match_layer = "ready_now"
    elif match_score >= 50 and len(gaps) <= 2:
        match_layer = "close_gap"
    elif opportunity.get("type") == "training_pathway":
        match_layer = "training_pathway"
    else:
        match_layer = "close_gap"

    return {
        "match_score": match_score,
        "gaps": gaps,
        "met": met,
        "isco_compatible": isco_compatible,
        "match_layer": match_layer
    }


def _format_dimension_summary(dimension_results: dict) -> str:
    lines = []
    for dim_id, result in dimension_results.items():
        tier = TIER_LABELS.get(result.get("tier_achieved", 1), "Entry")
        lines.append(f"- {result.get('dimension_label', dim_id)}: {tier} ({result.get('score', 0)}/100)")
    return "\n".join(lines)


def _format_required(required_dimensions: dict) -> str:
    lines = []
    for dim_id, tier in required_dimensions.items():
        lines.append(f"- {dim_id.replace('_', ' ').title()}: {TIER_LABELS.get(tier, 'Entry')} or above")
    return "\n".join(lines)


def _format_gaps(gaps: list) -> str:
    if not gaps:
        return "No gaps — full match"
    lines = []
    for g in gaps:
        lines.append(f"- {g['label']}: currently {g['actual_label']}, needs {g['required_label']}")
    return "\n".join(lines)


def _generate_explanation(
    opportunity: dict,
    match_result: dict,
    dimension_results: dict,
    occupation_title: str,
    education: str,
    experience_years: int,
    region: str
) -> str:
    """Generate a personalised, honest match explanation using GPT."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    match_layer_labels = {
        "ready_now": "Ready now — meets all requirements",
        "close_gap": "Close gap — 1-2 skills to develop",
        "training_pathway": "Training pathway — structured development"
    }

    prompt = MATCH_EXPLANATION_PROMPT.format(
        occupation_title=occupation_title,
        education=education,
        experience_years=experience_years,
        region=region,
        opportunity_title=opportunity["title"],
        opportunity_type=opportunity["type_label"],
        match_layer=match_layer_labels.get(match_result["match_layer"], ""),
        dimension_summary=_format_dimension_summary(dimension_results),
        required_dimensions=_format_required(opportunity.get("required_dimensions", {})),
        gaps=_format_gaps(match_result["gaps"])
    )

    message = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=200,
        messages=[
            {"role": "system", "content": "You write short, honest, human opportunity explanations for informal economy workers."},
            {"role": "user", "content": prompt}
        ]
    )

    return message.choices[0].message.content.strip()


def match_opportunities(
    dimension_results: dict,
    isco_code: str,
    region_id: str,
    occupation_title: str,
    education: str = "",
    experience_years: int = 0
) -> dict:
    """
    Pipeline 7: Match a worker's skill profile to local opportunities.

    Returns opportunities grouped into 3 honest layers:
    - ready_now: can pursue immediately
    - close_gap: 1-2 dimensions to develop
    - training_pathway: structured development routes
    """
    opportunities = _load_opportunities(region_id)
    meta = _load_meta(region_id)

    if not opportunities:
        return {"ready_now": [], "close_gap": [], "training_pathway": [], "meta": meta}

    isco_group = int(isco_code[0]) if isco_code else 5

    scored = []
    for opp in opportunities:
        match_result = _score_match(opp, dimension_results, isco_group)
        scored.append((opp, match_result))

    # Sort by match score descending
    scored.sort(key=lambda x: x[1]["match_score"], reverse=True)

    ready_now = []
    close_gap = []
    training_pathway = []

    for opp, match_result in scored:
        layer = match_result["match_layer"]

        # Training pathways always go to their own layer
        if opp.get("type") == "training_pathway":
            layer = "training_pathway"

        # Generate personalised explanation
        explanation = _generate_explanation(
            opportunity=opp,
            match_result=match_result,
            dimension_results=dimension_results,
            occupation_title=occupation_title,
            education=education,
            experience_years=experience_years,
            region=region_id
        )

        entry = {
            "id": opp["id"],
            "title": opp["title"],
            "type": opp["type"],
            "type_label": opp["type_label"],
            "match_score": match_result["match_score"],
            "match_layer": layer,
            "gaps": match_result["gaps"],
            "met_dimensions": match_result["met"],
            "explanation": explanation,
            "bridge": opp.get("bridge", {}),
            "tags": opp.get("tags", []),
            "honest_description": opp.get("honest_description", ""),
            "econometric_signals": opp.get("econometric_signals", {}),
            "data_sources": meta.get("econometric_sources", {})
        }

        if layer == "ready_now":
            ready_now.append(entry)
        elif layer == "training_pathway":
            training_pathway.append(entry)
        else:
            close_gap.append(entry)

    return {
        "ready_now": ready_now[:3],
        "close_gap": close_gap[:2],
        "training_pathway": training_pathway[:2],
        "econometric_sources": meta.get("econometric_sources", {}),
        "currency": meta.get("currency", ""),
        "total_matched": len(ready_now) + len(close_gap) + len(training_pathway)
    }
