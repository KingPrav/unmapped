"""
PIPELINE 5 — Adaptive Difficulty Controller
No external data source needed — pure logic layer.
Tracks assessment state per session and adjusts tier based on performance.
Replaces STEP calibration with adaptive difficulty (more robust, works for any country).
"""

from dataclasses import dataclass, field
from typing import Optional


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
            "score": evaluation_result["score"],
            "tier_achieved": evaluation_result["tier_achieved"],
            "passed": evaluation_result["passed"],
            "feedback": evaluation_result["feedback"],
            "employer_signal": evaluation_result["employer_signal"],
            "dimension_label": evaluation_result["dimension_label"]
        }
        self.current_dimension_index += 1

        # Adaptive tier: if worker passed well (score >= 70), bump tier for next
        if evaluation_result["score"] >= 70 and self.current_tier < 3:
            self.current_tier = min(3, self.current_tier + 1)
        elif evaluation_result["score"] < 40 and self.current_tier > 1:
            self.current_tier = max(1, self.current_tier - 1)

        if self.current_dimension_index >= len(self.dimensions_to_assess):
            self.completed = True

    def progress(self) -> dict:
        total = len(self.dimensions_to_assess)
        done = len(self.dimension_results)
        return {
            "completed_dimensions": done,
            "total_dimensions": total,
            "percent": int((done / total) * 100) if total > 0 else 0,
            "current_tier": self.current_tier,
            "is_complete": self.completed
        }


# In-memory session store (production: replace with Redis)
_sessions: dict[str, AssessmentSession] = {}


def create_session(
    session_id: str,
    occupation_title: str,
    isco_code: str,
    region_id: str,
    dimensions_to_assess: list
) -> AssessmentSession:
    session = AssessmentSession(
        session_id=session_id,
        occupation_title=occupation_title,
        isco_code=isco_code,
        region_id=region_id,
        dimensions_to_assess=dimensions_to_assess
    )
    _sessions[session_id] = session
    return session


def get_session(session_id: str) -> Optional[AssessmentSession]:
    return _sessions.get(session_id)


def record_answer(session_id: str, evaluation_result: dict) -> AssessmentSession:
    session = _sessions[session_id]
    session.advance(evaluation_result)
    return session
