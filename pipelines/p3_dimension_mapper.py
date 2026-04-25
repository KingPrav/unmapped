"""
PIPELINE 3 — Skill Dimension Mapper
Data source: dwa_dimension_map.json (pre-built lookup, grounded in O*NET DWA categories)
Takes list of DWAs → maps to 5 skill dimensions → returns assessment plan
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MAP_PATH = BASE_DIR / "data" / "seed" / "dwa_dimension_map.json"

with open(MAP_PATH) as f:
    DIMENSION_MAP = json.load(f)["dimensions"]

DIMENSION_ORDER = [
    "fault_diagnosis",
    "communication",
    "resource_judgment",
    "process_quality",
    "operational_organization"
]


def _keyword_match(dwa_text: str) -> str:
    """
    Classify a DWA text into a dimension using keyword matching.
    Used when a DWA doesn't have a pre-assigned dimension.
    """
    text_lower = dwa_text.lower()
    scores = {}
    for dim_id, dim_data in DIMENSION_MAP.items():
        score = sum(1 for kw in dim_data["keywords"] if kw in text_lower)
        scores[dim_id] = score

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "fault_diagnosis"  # default


def map_dimensions(dwas_by_dimension: dict, all_dwas: list) -> dict:
    """
    Pipeline 3: Maps DWAs to skill dimensions and builds assessment plan.

    Args:
        dwas_by_dimension: DWAs already grouped by dimension (from Pipeline 2)
        all_dwas: Full list of DWAs (for fallback keyword matching)

    Returns:
        dict with dimension_plan — which dimensions to assess and with which DWAs
    """
    dimension_plan = {}

    # Use pre-assigned dimensions from seed/O*NET data
    for dim_id in DIMENSION_ORDER:
        dim_meta = DIMENSION_MAP[dim_id]
        dwas_for_dim = dwas_by_dimension.get(dim_id, [])

        # If no pre-assigned DWAs for this dimension, try keyword matching
        if not dwas_for_dim:
            for dwa in all_dwas:
                if _keyword_match(dwa["text"]) == dim_id:
                    dwas_for_dim.append(dwa)

        if dwas_for_dim:
            dimension_plan[dim_id] = {
                "dimension_id": dim_id,
                "label": dim_meta["label"],
                "description": dim_meta["description"],
                "employer_signal": dim_meta["employer_signal"],
                "supporting_dwas": dwas_for_dim,
                "primary_dwa": dwas_for_dim[0]["text"]  # Best DWA for challenge generation
            }

    # Ensure we always have at least 3 dimensions for a meaningful assessment
    assessed_count = len(dimension_plan)
    if assessed_count < 3:
        for dim_id in DIMENSION_ORDER:
            if dim_id not in dimension_plan:
                dim_meta = DIMENSION_MAP[dim_id]
                dimension_plan[dim_id] = {
                    "dimension_id": dim_id,
                    "label": dim_meta["label"],
                    "description": dim_meta["description"],
                    "employer_signal": dim_meta["employer_signal"],
                    "supporting_dwas": [],
                    "primary_dwa": dim_meta["description"]
                }
            if len(dimension_plan) >= 5:
                break

    return {
        "dimension_plan": dimension_plan,
        "dimensions_to_assess": list(dimension_plan.keys()),
        "total_dimensions": len(dimension_plan)
    }
