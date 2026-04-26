"""
Shared scoring — ISCO-aware dimension weighting.
Imported by p6 (profile generation) and p7 (opportunity matching)
so both modules use the same weighted overall score.

Rationale:
  A phone repair technician (ISCO 7) should have fault_diagnosis count 40%
  of their score. A market seller (ISCO 5) should have communication count
  40%. A flat average of 5 dimensions misrepresents both.

Dimension IDs:
  fault_diagnosis        — identifying and fixing what's wrong
  communication          — interacting with customers, peers, employers
  resource_judgment      — managing materials, money, time
  process_quality        — following steps, maintaining standards
  operational_organization — planning, scheduling, organising work
"""

# ISCO-08 major group → dimension weight map.
# Each dict sums to 1.0.
ISCO_WEIGHTS: dict = {
    1: {  # Managers
        "fault_diagnosis":          0.05,
        "communication":            0.30,
        "resource_judgment":        0.25,
        "process_quality":          0.15,
        "operational_organization": 0.25,
    },
    2: {  # Professionals
        "fault_diagnosis":          0.30,
        "communication":            0.25,
        "resource_judgment":        0.10,
        "process_quality":          0.30,
        "operational_organization": 0.05,
    },
    3: {  # Technicians & Associate Professionals
        "fault_diagnosis":          0.35,
        "communication":            0.20,
        "resource_judgment":        0.05,
        "process_quality":          0.25,
        "operational_organization": 0.15,
    },
    4: {  # Clerical Support Workers
        "fault_diagnosis":          0.05,
        "communication":            0.35,
        "resource_judgment":        0.10,
        "process_quality":          0.20,
        "operational_organization": 0.30,
    },
    5: {  # Service & Sales Workers
        "fault_diagnosis":          0.05,
        "communication":            0.40,
        "resource_judgment":        0.25,
        "process_quality":          0.10,
        "operational_organization": 0.20,
    },
    6: {  # Agricultural Workers
        "fault_diagnosis":          0.20,
        "communication":            0.05,
        "resource_judgment":        0.35,
        "process_quality":          0.25,
        "operational_organization": 0.15,
    },
    7: {  # Craft & Related Trades (phone repair, mechanics, tailors, welders…)
        "fault_diagnosis":          0.40,
        "communication":            0.05,
        "resource_judgment":        0.15,
        "process_quality":          0.30,
        "operational_organization": 0.10,
    },
    8: {  # Plant & Machine Operators / Assemblers
        "fault_diagnosis":          0.35,
        "communication":            0.05,
        "resource_judgment":        0.10,
        "process_quality":          0.30,
        "operational_organization": 0.20,
    },
    9: {  # Elementary Occupations
        "fault_diagnosis":          0.05,
        "communication":            0.15,
        "resource_judgment":        0.30,
        "process_quality":          0.25,
        "operational_organization": 0.25,
    },
}

# Equal fallback — used when ISCO code is absent or unrecognised
DEFAULT_WEIGHTS: dict = {
    "fault_diagnosis":          0.20,
    "communication":            0.20,
    "resource_judgment":        0.20,
    "process_quality":          0.20,
    "operational_organization": 0.20,
}


def compute_weighted_score(dimension_results: dict, isco_code: str = "") -> int:
    """
    Compute overall score using ISCO-group-specific dimension weights.

    Normalises automatically when not all 5 dimensions were assessed,
    so partial assessments still produce a meaningful score — the
    unassessed dimensions simply don't contribute to the denominator.

    Args:
        dimension_results: {dim_id: {"score": int, ...}} from session
        isco_code:         e.g. "7421" → uses ISCO group 7 weights

    Returns:
        Weighted integer score 0–100
    """
    if not dimension_results:
        return 0

    isco_group = int(isco_code[0]) if isco_code and isco_code[0].isdigit() else 0
    weights = ISCO_WEIGHTS.get(isco_group, DEFAULT_WEIGHTS)

    weighted_sum = 0.0
    total_weight  = 0.0

    for dim_id, result in dimension_results.items():
        w = weights.get(dim_id, 0.20)
        weighted_sum += result.get("score", 0) * w
        total_weight  += w

    return int(weighted_sum / total_weight) if total_weight > 0 else 0


def get_dimension_weights(isco_code: str) -> dict:
    """
    Return the weight map for a given ISCO code.
    Used by the UI to show why certain dimensions matter more for the worker's occupation.
    """
    isco_group = int(isco_code[0]) if isco_code and isco_code[0].isdigit() else 0
    return ISCO_WEIGHTS.get(isco_group, DEFAULT_WEIGHTS)
