"""
PIPELINE 5 — Adaptive Difficulty Controller
Tracks assessment state per session and adjusts tier based on performance.

Persistence: sessions are stored in SQLite (database/db.py) so data survives
server restarts. The in-memory _sessions dict is the live cache; SQLite is the
source of truth that repopulates it on startup.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Ensure project root is on path for database import
sys.path.append(str(Path(__file__).parent.parent))
from database.db import init_db, save_session as _db_save, load_all_sessions


# ─────────────────────────────────────────────
# Session dataclass
# ─────────────────────────────────────────────

@dataclass
class AssessmentSession:
    session_id: str
    occupation_title: str
    isco_code: str
    region_id: str
    dimensions_to_assess: list
    dimension_results: dict = field(default_factory=dict)
    current_dimension_index: int = 0
    current_tier: int = 1
    completed: bool = False

    def current_dimension(self) -> Optional[str]:
        if self.current_dimension_index < len(self.dimensions_to_assess):
            return self.dimensions_to_assess[self.current_dimension_index]
        return None

    def advance(self, evaluation_result: dict):
        """Record result and move to next dimension."""
        dim_id = evaluation_result["dimension_id"]
        self.dimension_results[dim_id] = {
            "score":            evaluation_result["score"],
            "tier_achieved":    evaluation_result["tier_achieved"],
            "passed":           evaluation_result["passed"],
            "feedback":         evaluation_result["feedback"],
            "employer_signal":  evaluation_result["employer_signal"],
            "dimension_label":  evaluation_result["dimension_label"]
        }
        self.current_dimension_index += 1

        # Adaptive tier: good score bumps difficulty, poor score drops it
        if evaluation_result["score"] >= 70 and self.current_tier < 3:
            self.current_tier = min(3, self.current_tier + 1)
        elif evaluation_result["score"] < 40 and self.current_tier > 1:
            self.current_tier = max(1, self.current_tier - 1)

        if self.current_dimension_index >= len(self.dimensions_to_assess):
            self.completed = True

    def progress(self) -> dict:
        total = len(self.dimensions_to_assess)
        done  = len(self.dimension_results)
        return {
            "completed_dimensions": done,
            "total_dimensions":     total,
            "percent":              int((done / total) * 100) if total > 0 else 0,
            "current_tier":         self.current_tier,
            "is_complete":          self.completed
        }


# ─────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────

def _serialize(session: AssessmentSession) -> dict:
    """Flatten session + all dynamic attributes into a JSON-safe dict."""
    return {
        "session_id":              session.session_id,
        "occupation_title":        session.occupation_title,
        "isco_code":               session.isco_code,
        "region_id":               session.region_id,
        "dimensions_to_assess":    session.dimensions_to_assess,
        "dimension_results":       session.dimension_results,
        "current_dimension_index": session.current_dimension_index,
        "current_tier":            session.current_tier,
        "completed":               session.completed,
        # Dynamic attributes set by api/main.py after creation
        "_dimension_plan":  getattr(session, "_dimension_plan",  {}),
        "_config":          getattr(session, "_config",          {}),
        "_education_level": getattr(session, "_education_level", "upper_secondary"),
        "_experience_years":getattr(session, "_experience_years", 0),
        "_other_skills":    getattr(session, "_other_skills",    ""),
        "_profile_id":      getattr(session, "_profile_id",      None),
        "_current_challenge":getattr(session,"_current_challenge",None),
    }


def _deserialize(data: dict) -> AssessmentSession:
    """Reconstruct an AssessmentSession from a serialised dict."""
    session = AssessmentSession(
        session_id              = data["session_id"],
        occupation_title        = data["occupation_title"],
        isco_code               = data["isco_code"],
        region_id               = data["region_id"],
        dimensions_to_assess    = data["dimensions_to_assess"],
        dimension_results       = data.get("dimension_results", {}),
        current_dimension_index = data.get("current_dimension_index", 0),
        current_tier            = data.get("current_tier", 1),
        completed               = data.get("completed", False),
    )
    session._dimension_plan   = data.get("_dimension_plan", {})
    session._config           = data.get("_config", {})
    session._education_level  = data.get("_education_level", "upper_secondary")
    session._experience_years = data.get("_experience_years", 0)
    session._other_skills     = data.get("_other_skills", "")
    session._profile_id       = data.get("_profile_id", None)
    session._current_challenge= data.get("_current_challenge", None)
    return session


def _persist(session: AssessmentSession):
    """Write current session state to SQLite."""
    _db_save(
        session_id = session.session_id,
        region_id  = session.region_id,
        completed  = session.completed,
        data       = _serialize(session)
    )


# ─────────────────────────────────────────────
# In-memory cache — repopulated from SQLite on startup
# ─────────────────────────────────────────────

# Initialise DB schema
init_db()

# Load all existing sessions into memory
_sessions: dict[str, AssessmentSession] = {}
for _raw in load_all_sessions():
    try:
        _s = _deserialize(_raw)
        _sessions[_s.session_id] = _s
    except Exception:
        pass  # Skip corrupt rows — don't crash startup


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def create_session(
    session_id: str,
    occupation_title: str,
    isco_code: str,
    region_id: str,
    dimensions_to_assess: list
) -> AssessmentSession:
    session = AssessmentSession(
        session_id           = session_id,
        occupation_title     = occupation_title,
        isco_code            = isco_code,
        region_id            = region_id,
        dimensions_to_assess = dimensions_to_assess
    )
    _sessions[session_id] = session
    _persist(session)
    return session


def get_session(session_id: str) -> Optional[AssessmentSession]:
    return _sessions.get(session_id)


def record_answer(session_id: str, evaluation_result: dict) -> AssessmentSession:
    session = _sessions[session_id]
    session.advance(evaluation_result)
    _persist(session)
    return session


def persist_session(session_id: str):
    """
    Explicit save — call this after setting dynamic attributes on a session
    (e.g. _dimension_plan, _config, _profile_id) that happen outside record_answer.
    """
    session = _sessions.get(session_id)
    if session:
        _persist(session)
