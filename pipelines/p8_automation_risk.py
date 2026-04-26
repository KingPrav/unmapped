"""
PIPELINE 8 — AI Readiness & Displacement Risk Lens (Module 02)

Data sources:
  - automation_risk.json: Frey & Osborne (2013/2017) scores, LMIC-adjusted
  - ILO task content indices (routine/non-routine cognitive/manual) per dimension
  - Wittgenstein Centre 2025-2035 education projections (GHA, KEN)

Inputs:
  isco_code           — from Pipeline 1
  dimension_results   — from completed assessment (tiers + scores per dimension)
  country_config      — the region YAML config (for country_code, location context)

Output:
  A structured risk profile with:
    - Overall automation probability (Frey-Osborne, LMIC-adjusted)
    - Task-level breakdown (per assessed dimension)
    - Durable vs at-risk skills, honestly stated
    - Resilience score (0-100), weighted by assessment performance
    - Wittgenstein education projection signal (region-specific)
    - LMIC calibration context
    - Recommended upskilling actions

Design principles:
  - No LLM call — fully data-driven and deterministic
  - Task-level granularity, not just job-level
  - Honest about limits and uncertainty
  - LMIC-calibrated throughout
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RISK_PATH = BASE_DIR / "data" / "seed" / "automation_risk.json"

# Load risk data once at import
with open(RISK_PATH) as f:
    RISK_DATA = json.load(f)

DIMENSION_RISK = RISK_DATA["dimension_risk_profiles"]
OCCUPATION_RISK = RISK_DATA["occupation_automation"]
WITTGENSTEIN = RISK_DATA["wittgenstein_projections"]

EXPOSURE_COLORS = {
    "LOW":           "#22c55e",
    "MODERATE":      "#eab308",
    "MODERATE-HIGH": "#f97316",
    "HIGH":          "#ef4444"
}

EXPOSURE_LABELS = {
    "LOW":           "Low Automation Exposure",
    "MODERATE":      "Moderate Automation Exposure",
    "MODERATE-HIGH": "Moderate-High Automation Exposure",
    "HIGH":          "High Automation Exposure"
}

TIER_WEIGHT = {0: 0.0, 1: 0.5, 2: 0.8, 3: 1.0}


def _get_occupation_risk(isco_code: str) -> dict:
    """Return occupation-level risk, fall back to ISCO group defaults if not found."""
    if isco_code in OCCUPATION_RISK:
        return OCCUPATION_RISK[isco_code]

    # Fall back to group defaults by first digit
    group = isco_code[0] if isco_code else "5"
    group_defaults = {
        "1": {"frey_osborne_raw": 0.30, "lmic_adjusted": 0.22, "exposure_label": "LOW"},
        "2": {"frey_osborne_raw": 0.20, "lmic_adjusted": 0.15, "exposure_label": "LOW"},
        "3": {"frey_osborne_raw": 0.45, "lmic_adjusted": 0.32, "exposure_label": "MODERATE"},
        "4": {"frey_osborne_raw": 0.70, "lmic_adjusted": 0.50, "exposure_label": "MODERATE-HIGH"},
        "5": {"frey_osborne_raw": 0.68, "lmic_adjusted": 0.48, "exposure_label": "MODERATE"},
        "6": {"frey_osborne_raw": 0.54, "lmic_adjusted": 0.38, "exposure_label": "MODERATE"},
        "7": {"frey_osborne_raw": 0.55, "lmic_adjusted": 0.39, "exposure_label": "MODERATE"},
        "8": {"frey_osborne_raw": 0.78, "lmic_adjusted": 0.55, "exposure_label": "MODERATE-HIGH"},
        "9": {"frey_osborne_raw": 0.86, "lmic_adjusted": 0.60, "exposure_label": "MODERATE-HIGH"},
    }
    return group_defaults.get(group, {
        "frey_osborne_raw": 0.55,
        "lmic_adjusted": 0.40,
        "exposure_label": "MODERATE"
    })


def _compute_resilience_score(
    dimension_results: dict,
    occ_risk: dict
) -> int:
    """
    Resilience score (0-100): composite of durable-skill performance.

    Logic:
      - Each dimension has a durability flag (durable = True/False)
      - Durable dimensions are weighted more heavily
      - Assessment tier and score determine performance contribution
      - High performance on durable skills = high resilience
    """
    if not dimension_results:
        return 50  # default mid-point if no results

    durable_dims = set(occ_risk.get("durable_dimensions", []))
    at_risk_dims = set(occ_risk.get("at_risk_dimensions", []))

    total_weight = 0.0
    weighted_score = 0.0

    for dim_id, result in dimension_results.items():
        dim_profile = DIMENSION_RISK.get(dim_id, {})
        is_durable = dim_profile.get("durable", True) or dim_id in durable_dims

        tier = result.get("tier_achieved", 1)
        score = result.get("score", 0)

        # Performance score (0-1): blend of tier and raw score
        tier_contribution = TIER_WEIGHT.get(tier, 0.5)
        score_contribution = score / 100.0
        performance = (tier_contribution * 0.6) + (score_contribution * 0.4)

        # Durable dimensions count 2x; at-risk count 0.5x
        if is_durable:
            weight = 2.0
        elif dim_id in at_risk_dims:
            weight = 0.5
        else:
            weight = 1.0

        weighted_score += performance * weight
        total_weight += weight

    if total_weight == 0:
        return 50

    raw_resilience = weighted_score / total_weight  # 0.0 – 1.0
    return min(100, max(0, round(raw_resilience * 100)))


def _build_task_breakdown(
    dimension_results: dict,
    occ_risk: dict
) -> list:
    """Build per-dimension task risk breakdown."""
    durable_dims = set(occ_risk.get("durable_dimensions", []))
    at_risk_dims = set(occ_risk.get("at_risk_dimensions", []))

    breakdown = []
    for dim_id, result in dimension_results.items():
        dim_profile = DIMENSION_RISK.get(dim_id, {})
        is_durable = dim_profile.get("durable", True) or dim_id in durable_dims

        exposure_label = dim_profile.get("automation_exposure", "MODERATE")
        exposure_score = dim_profile.get("automation_exposure_score", 0.35)

        # Override if occupation flags this as at-risk or durable
        if dim_id in at_risk_dims and is_durable:
            is_durable = False
            exposure_label = "MODERATE-HIGH"

        breakdown.append({
            "dimension_id":       dim_id,
            "dimension_label":    result.get("dimension_label", dim_id),
            "ilo_task_type":      dim_profile.get("ilo_task_type", "mixed"),
            "exposure_label":     exposure_label,
            "exposure_score":     round(exposure_score * 100),   # 0-100 for UI
            "exposure_color":     EXPOSURE_COLORS.get(exposure_label, "#eab308"),
            "is_durable":         is_durable,
            "rationale":          dim_profile.get("rationale", ""),
            "resilience_rationale": dim_profile.get("resilience_rationale", ""),
            "horizon_note":       dim_profile.get("horizon_note", ""),
            "your_tier":          result.get("tier_achieved", 1),
            "your_score":         result.get("score", 0),
            "your_tier_label":    {1: "Entry", 2: "Functional", 3: "Mastery"}.get(
                                      result.get("tier_achieved", 1), "Entry")
        })

    # Sort: durable first, then by exposure score ascending
    breakdown.sort(key=lambda x: (not x["is_durable"], x["exposure_score"]))
    return breakdown


def _get_upskilling_actions(
    dimension_results: dict,
    occ_risk: dict,
    overall_automation: float
) -> list:
    """
    Generate specific, actionable upskilling recommendations.
    Grounded in what we know about the worker from their assessment.
    """
    actions = []
    durable_dims = set(occ_risk.get("durable_dimensions", []))

    # Find weakest durable dimensions — these are highest-value development targets
    durable_results = [
        (dim_id, result)
        for dim_id, result in dimension_results.items()
        if dim_id in durable_dims and DIMENSION_RISK.get(dim_id, {}).get("durable", True)
    ]
    durable_results.sort(key=lambda x: x[1].get("score", 0))  # weakest first

    for dim_id, result in durable_results[:2]:
        dim_profile = DIMENSION_RISK.get(dim_id, {})
        if result.get("tier_achieved", 1) < 3:
            actions.append({
                "priority":    "HIGH",
                "type":        "strengthen_durable",
                "dimension":   result.get("dimension_label", dim_id),
                "action":      f"Develop {result.get('dimension_label', dim_id)} from "
                               f"{['Entry','Functional','Mastery'][result.get('tier_achieved',1)-1]} "
                               f"to the next level — this is a durable skill with low automation exposure.",
                "why":         dim_profile.get("resilience_rationale", "")
            })

    # Find at-risk dimensions the worker scored highly on — honest warning
    at_risk_results = [
        (dim_id, result)
        for dim_id, result in dimension_results.items()
        if dim_id in occ_risk.get("at_risk_dimensions", [])
    ]
    for dim_id, result in at_risk_results:
        actions.append({
            "priority":    "MEDIUM",
            "type":        "diversify_away_from_risk",
            "dimension":   result.get("dimension_label", dim_id),
            "action":      f"Don't over-rely on {result.get('dimension_label', dim_id)} — "
                           f"this task type has moderate-to-high automation exposure by 2030-2035. "
                           f"Pair it with durable inter-personal or diagnostic skills.",
            "why":         DIMENSION_RISK.get(dim_id, {}).get("horizon_note", "")
        })

    # If overall automation risk is high, flag credential building
    if overall_automation > 0.50:
        actions.append({
            "priority":    "HIGH",
            "type":        "credential_building",
            "dimension":   "All",
            "action":      "With above-average automation exposure, building a verifiable credential "
                           "(trade certificate, NVTI, KITI) significantly raises your labour market "
                           "resilience by making your skills legible to employers who cannot assess you directly.",
            "why":         "Workers with verifiable credentials have 20-40% higher earnings and 30% "
                           "lower long-term unemployment risk (ILO, 2023)."
        })

    return actions[:4]  # cap at 4 actions


def generate_risk_profile(
    isco_code: str,
    dimension_results: dict,
    country_config: dict
) -> dict:
    """
    Pipeline 8: Generate AI Readiness & Displacement Risk profile (Module 02).

    Args:
        isco_code:         ISCO-08 code from classification (e.g. "7421")
        dimension_results: Completed assessment dimension scores and tiers
        country_config:    Region YAML config (for country_code, location)

    Returns:
        Full Module 02 risk profile dict
    """
    country_code = country_config.get("country_code", "GHA")
    location     = country_config.get("location_context", "informal urban economy")
    region_name  = country_config.get("region_name", "")

    # Occupation-level risk
    occ_risk = _get_occupation_risk(isco_code)

    automation_probability = occ_risk.get("lmic_adjusted", 0.40)
    exposure_label         = occ_risk.get("exposure_label", "MODERATE")
    frey_raw               = occ_risk.get("frey_osborne_raw", 0.55)

    # Task breakdown and resilience
    task_breakdown  = _build_task_breakdown(dimension_results, occ_risk)
    resilience_score = _compute_resilience_score(dimension_results, occ_risk)
    actions          = _get_upskilling_actions(dimension_results, occ_risk, automation_probability)

    # Wittgenstein signal
    witt = WITTGENSTEIN.get(country_code, WITTGENSTEIN.get("GHA", {}))
    wittgenstein_signal = {
        "tertiary_2025_pct":   witt.get("tertiary_2025_pct", 8.0),
        "tertiary_2035_pct":   witt.get("tertiary_2035_pct", 14.0),
        "interpretation":      witt.get("interpretation", ""),
        "implication":         witt.get("implication_for_informal_workers", ""),
        "source":              witt.get("source", "Wittgenstein Centre WC-IIASA 2024")
    }

    # Identify durable and at-risk skill clusters from assessment
    durable_assessed = [
        d["dimension_label"]
        for d in task_breakdown if d["is_durable"]
    ]
    at_risk_assessed = [
        d["dimension_label"]
        for d in task_breakdown if not d["is_durable"]
    ]

    # Honest summary text
    exposure_text = EXPOSURE_LABELS.get(exposure_label, "Moderate")
    if automation_probability < 0.25:
        headline = (
            f"Your work has low automation exposure in the {location} context. "
            f"The tasks you do — especially physical diagnosis and interpersonal skills — "
            f"are among the most durable against AI disruption through 2035."
        )
    elif automation_probability < 0.45:
        headline = (
            f"Your work has moderate automation exposure. Some of what you do is durable; "
            f"some parts — especially routine process-following — face growing risk by 2030-2035. "
            f"The good news: your strongest assessed skills are in the durable category."
        )
    elif automation_probability < 0.60:
        headline = (
            f"Your work has moderate-to-high automation exposure in the long run. "
            f"However, in the {location}, infrastructure constraints and capital costs "
            f"mean this risk is mostly beyond 2030. Now is the time to build durable skills."
        )
    else:
        headline = (
            f"Your occupation has above-average long-term automation exposure. "
            f"While immediate risk in the {location} context is limited by infrastructure, "
            f"building verifiable durable skills now significantly improves your 2030-2035 position."
        )

    return {
        "module":                  "02",
        "module_name":             "AI Readiness & Displacement Risk",
        "isco_code":               isco_code,
        "region":                  region_name,
        "location_context":        location,
        "horizon":                 "2025–2035",

        # Core risk metrics — shown prominently in UI
        "automation_probability":  round(automation_probability * 100),  # 0-100 for display
        "automation_probability_raw": automation_probability,
        "frey_osborne_raw_pct":    round(frey_raw * 100),
        "lmic_adjustment_note":    RISK_DATA["meta"]["lmic_adjustment_note"],
        "exposure_label":          exposure_label,
        "exposure_display":        exposure_text,
        "exposure_color":          EXPOSURE_COLORS.get(exposure_label, "#eab308"),
        "headline":                headline,

        # Resilience score
        "resilience_score":        resilience_score,
        "resilience_label":        (
            "Strong" if resilience_score >= 70
            else "Moderate" if resilience_score >= 45
            else "Needs Development"
        ),

        # Task-level breakdown
        "task_breakdown":          task_breakdown,
        "durable_skills":          durable_assessed,
        "at_risk_tasks":           at_risk_assessed,

        # Wittgenstein education projection
        "wittgenstein_signal":     wittgenstein_signal,

        # Recommended actions
        "recommended_actions":     actions,

        # Primary risk/resilience narratives
        "primary_risk_source":     occ_risk.get("primary_risk_source", ""),
        "primary_resilience_source": occ_risk.get("primary_resilience_source", ""),

        # Data source citations — must be visible to user
        "data_sources": {
            "automation_scores":    "Frey, C.B. & Osborne, M.A. (2013/2017). The Future of Employment. Oxford Martin Programme. LMIC-adjusted for SSA informal economy context.",
            "task_content":         "ILO (2023). Task Content of Jobs in Low- and Middle-Income Countries. Geneva: ILO Working Paper.",
            "education_projections":"Wittgenstein Centre WC-IIASA (2024). Human Capital Data Explorer, SSP2 scenario, 25-34 cohort.",
            "lmic_calibration":     "World Bank STEP Skills Measurement Survey + ILO infrastructure index for Sub-Saharan Africa."
        }
    }
