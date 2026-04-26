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

MATCH_EXPLANATION_PROMPT = """You are writing a short, brutally honest explanation of why a specific opportunity fits (or does not yet fit) a worker's skill profile.

Worker:
- Occupation: {occupation_title}
- Education: {education}
- Experience: {experience_years} years
- Region: {region}
- Overall skill score: {overall_score}/100

Opportunity: {opportunity_title} ({opportunity_type})
Match layer: {match_layer}

Their skill assessment:
{dimension_summary}

Required for this opportunity:
{required_dimensions}

Gaps (if any):
{gaps}

CRITICAL RULES — follow these strictly:
- If overall score is below 40: do NOT say they are ready or close. Be clear that skill development is needed first. Focus on what to build, not what to pursue now.
- If match layer is "close_gap": name the specific gap plainly and give a concrete development action.
- If match layer is "ready_now": confirm what makes them qualified, but still note any weaker areas honestly.
- If match layer is "training_pathway": explain why this structured route is the right next step for where they are now.
- Never sugarcoat a weak profile. Do not say things like "your communication skills open doors" if their communication score is low.
- Do NOT mention the assessment system by name.
- Write directly to the person. 2-3 sentences maximum. Be concise and human."""


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


def _compute_overall_score(dimension_results: dict) -> int:
    """Compute overall score as average across all assessed dimensions."""
    scores = [r.get("score", 0) for r in dimension_results.values()]
    return int(sum(scores) / len(scores)) if scores else 0


def _score_match(opportunity: dict, dimension_results: dict, isco_group: int, overall_score: int) -> dict:
    """
    Score how well a worker's profile matches an opportunity.

    Honesty rules:
    - Score quality floor: a dimension score < 35 is treated as one tier below
      actual — passing a challenge at 10/100 is not the same as Entry-level competence.
    - ISCO matching: exact group match required for ready_now; adjacent only for close_gap.
    - Overall score floors:
        < 30 → training_pathway only (not ready for self-employment or employment)
        30–49 → close_gap at best, only if strong ISCO alignment
        50+ → normal layer assignment
    """
    required = opportunity.get("required_dimensions", {})
    opp_isco_groups = opportunity.get("isco_groups", [])

    # Strict vs soft ISCO compatibility
    isco_exact = isco_group in opp_isco_groups
    isco_adjacent = any(abs(isco_group - g) <= 1 for g in opp_isco_groups)
    isco_compatible = isco_exact or isco_adjacent

    if not required:
        # Training pathways — always available regardless of score
        return {
            "match_score": 60,
            "gaps": [],
            "met": [],
            "isco_exact": True,
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
        actual_score = actual.get("score", 0)

        # Score quality floor: low score within a tier counts as the tier below.
        # Scoring 8/100 on a tier-1 challenge is not Entry competence.
        effective_tier = actual_tier
        if actual_score < 35 and actual_tier > 0:
            effective_tier = actual_tier - 1

        if effective_tier >= required_tier:
            met_count += 1
            met.append({
                "dimension": dim_id,
                "required": required_tier,
                "actual": actual_tier,
                "label": actual.get("dimension_label", dim_id)
            })
        else:
            gap_size = required_tier - effective_tier
            gaps.append({
                "dimension": dim_id,
                "required": required_tier,
                "required_label": TIER_LABELS.get(required_tier, "Entry"),
                "actual": effective_tier,
                "actual_label": TIER_LABELS.get(effective_tier, "Below Entry"),
                "label": actual.get("dimension_label", dim_id),
                "gap_size": gap_size
            })

    match_pct = met_count / total_requirements if total_requirements > 0 else 0
    match_score = int(match_pct * 100)

    # ── Layer assignment — honest floors ──────────────────────────────────────
    if opportunity.get("type") == "training_pathway":
        match_layer = "training_pathway"

    elif overall_score < 30:
        # Score too low to recommend employment or self-employment of any kind.
        # Training is the only honest recommendation.
        match_layer = "training_pathway"

    elif overall_score < 50:
        # Marginal — only close_gap if skill match is strong AND occupation is relevant.
        if match_score >= 80 and isco_exact:
            match_layer = "close_gap"
        else:
            match_layer = "training_pathway"

    else:
        # Overall score is meaningful — use skill match + ISCO to determine layer.
        if match_score >= 100 and isco_exact:
            match_layer = "ready_now"
        elif match_score >= 100 and isco_adjacent:
            # Skills meet requirements but occupation is a stretch — honest close_gap.
            match_layer = "close_gap"
        elif match_score >= 60 and len(gaps) <= 2 and isco_compatible:
            match_layer = "close_gap"
        else:
            match_layer = "training_pathway"

    return {
        "match_score": match_score,
        "gaps": gaps,
        "met": met,
        "isco_exact": isco_exact,
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
    region: str,
    overall_score: int
) -> str:
    """Generate a personalised, honest match explanation using GPT."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    match_layer_labels = {
        "ready_now": "Ready now — meets all requirements",
        "close_gap": "Close gap — specific skills to develop before pursuing",
        "training_pathway": "Training pathway — skill development needed first"
    }

    prompt = MATCH_EXPLANATION_PROMPT.format(
        occupation_title=occupation_title,
        education=education,
        experience_years=experience_years,
        region=region,
        overall_score=overall_score,
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
            {"role": "system", "content": "You write short, brutally honest opportunity explanations for informal economy workers. Never give false hope. Never recommend opportunities that someone is not ready for. Be direct."},
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

    # Compute overall score once — used for honest layer floors
    overall_score = _compute_overall_score(dimension_results)

    scored = []
    for opp in opportunities:
        match_result = _score_match(opp, dimension_results, isco_group, overall_score)
        scored.append((opp, match_result))

    # Sort by match score descending
    scored.sort(key=lambda x: x[1]["match_score"], reverse=True)

    ready_now = []
    close_gap = []
    training_pathway = []

    for opp, match_result in scored:
        layer = match_result["match_layer"]

        # Training pathways always go to their own layer regardless of scoring
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
            region=region_id,
            overall_score=overall_score
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
        "overall_score": overall_score,
        "total_matched": len(ready_now) + len(close_gap) + len(training_pathway)
    }
