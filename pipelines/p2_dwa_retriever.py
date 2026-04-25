"""
PIPELINE 2 — DWA Retriever
Data source: O*NET Detailed Work Activities via ISCO-08 crosswalk
Takes ISCO-08 code → returns list of Detailed Work Activities for that occupation
Falls back to seed data if real O*NET data not yet loaded
"""

import json
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SEED_PATH = BASE_DIR / "data" / "seed" / "seed_data.json"
DB_PATH = BASE_DIR / "data" / "processed" / "occupation_index.db"

with open(SEED_PATH) as f:
    SEED_DATA = json.load(f)


def _get_dwas_from_db(isco_code: str) -> list:
    """Query real O*NET data from SQLite if available."""
    if not DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT dwa_id, dwa_text, dimension
            FROM dwas
            WHERE isco_code = ?
        """, (isco_code,))
        rows = cursor.fetchall()
        conn.close()
        return [{"id": r[0], "text": r[1], "dimension": r[2]} for r in rows]
    except Exception:
        return []


def _get_dwas_from_seed(isco_code: str) -> list:
    """Fall back to seed data."""
    occ = SEED_DATA["occupations"].get(isco_code)
    if occ:
        return occ["dwas"]
    return []


def retrieve_dwas(isco_code: str) -> dict:
    """
    Pipeline 2: Retrieves Detailed Work Activities for an ISCO-08 occupation.

    Args:
        isco_code: ISCO-08 unit group code (e.g. "7421")

    Returns:
        dict with dwas list and data_source indicator
    """
    # Try real data first, fall back to seed
    dwas = _get_dwas_from_db(isco_code)
    source = "onet_processed"

    if not dwas:
        dwas = _get_dwas_from_seed(isco_code)
        source = "seed_onet"

    # Group DWAs by dimension for downstream use
    by_dimension = {}
    for dwa in dwas:
        dim = dwa.get("dimension", "uncategorized")
        if dim not in by_dimension:
            by_dimension[dim] = []
        by_dimension[dim].append(dwa)

    return {
        "isco_code": isco_code,
        "dwas": dwas,
        "by_dimension": by_dimension,
        "total_dwas": len(dwas),
        "data_source": source
    }
