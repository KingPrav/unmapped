"""
UNMAPPED — FastAPI Backend
Wires all 7 pipelines into a clean REST API.

Endpoints:
  GET  /regions                         — List available country configs
  POST /assess/start                    — Start assessment (classify occupation)
  POST /assess/answer                   — Submit answer, get next challenge
  GET  /assess/profile/{session_id}     — Get final skill profile
  GET  /opportunities/{session_id}      — Get matched opportunities (Module 3)
  GET  /dashboard/{region}             — Policymaker aggregate signals (Module 3)
  GET  /workers/{region}               — Policymaker worker registry, ranked + filterable
  GET  /health                          — Health check
"""

import os
import uuid
import yaml
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Validate API key on startup
if not os.environ.get("OPENAI_API_KEY"):
    print("\n⚠️  WARNING: OPENAI_API_KEY not found in environment.")
    print("   Create a .env file in the project root with:")
    print("   OPENAI_API_KEY=sk-proj-your-key-here\n")

# Import all pipelines
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.p1_classifier import classify_occupation
from pipelines.p2_dwa_retriever import retrieve_dwas
from pipelines.p3_dimension_mapper import map_dimensions
from pipelines.p4_challenge_generator import generate_challenge, score_mcq_answer
from pipelines.p5_difficulty import create_session, get_session, record_answer, persist_session
from database.db import save_profile, get_profile as db_get_profile, profile_exists
from pipelines.p6_localiser import generate_profile
from pipelines.p7_opportunity_matcher import match_opportunities

BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
UI_DIR = BASE_DIR / "ui"

app = FastAPI(
    title="UNMAPPED API",
    description="Skills visibility infrastructure for informal economy workers",
    version="1.0.0"
)

# ─────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────

def load_config(region_id: str) -> dict:
    config_path = CONFIG_DIR / f"{region_id}.yaml"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Region config '{region_id}' not found")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class StartRequest(BaseModel):
    description: str                        # "I fix phones"
    region: str = "ghana_urban"             # config file name without .yaml
    education_level: str = "upper_secondary"
    experience_years: int = 0
    other_skills: str = ""

class AnswerRequest(BaseModel):
    session_id: str
    selected_option: str   # "A", "B", "C", or "D"


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "system": "UNMAPPED v1.0"}


@app.get("/regions")
def list_regions():
    """List all available country/region configurations."""
    configs = []
    for f in CONFIG_DIR.glob("*.yaml"):
        with open(f) as fp:
            cfg = yaml.safe_load(fp)
        configs.append({
            "id": cfg["region_id"],
            "name": cfg["region_name"],
            "country": cfg["country"],
            "language": cfg["language_name"]
        })
    return {"regions": configs}


@app.post("/assess/start")
def start_assessment(req: StartRequest):
    """
    Step 1: Classify occupation and generate first challenge.
    Runs Pipeline 1 → 2 → 3 → 4
    """
    config = load_config(req.region)

    # P1: Classify occupation
    try:
        classification = classify_occupation(req.description)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

    if not classification.get("occupation_data"):
        raise HTTPException(status_code=422, detail="Could not classify occupation. Try describing your work differently.")

    isco_code = classification["isco_code"]
    occupation_title = classification["title"]

    # P2: Retrieve DWAs
    dwa_result = retrieve_dwas(isco_code)

    # P3: Map to skill dimensions
    dimension_result = map_dimensions(
        dwa_result["by_dimension"],
        dwa_result["dwas"]
    )

    dimensions_to_assess = dimension_result["dimensions_to_assess"]

    # P5: Create session
    session_id = str(uuid.uuid4())
    session = create_session(
        session_id=session_id,
        occupation_title=occupation_title,
        isco_code=isco_code,
        region_id=req.region,
        dimensions_to_assess=dimensions_to_assess
    )

    # Store dimension plan, config, and user context in session
    session._dimension_plan   = dimension_result["dimension_plan"]
    session._config           = config
    session._education_level  = req.education_level
    session._experience_years = req.experience_years
    session._other_skills     = req.other_skills
    # 2-question-per-dimension state — initialise fresh
    session._current_q_num    = 1
    session._first_scores     = {}
    persist_session(session_id)   # flush dynamic attrs to SQLite

    # P4: Generate first challenge (Q1 of 2 for the first dimension)
    first_dim_id = session.current_dimension()
    first_dimension = dimension_result["dimension_plan"][first_dim_id]

    challenge_bundle = generate_challenge(
        occupation_title=occupation_title,
        dimension=first_dimension,
        country_config=config,
        tier=session.current_tier
    )

    # Store full challenge (with answer keys) server-side; send sanitised version to client
    session._current_challenge = challenge_bundle["full"]
    persist_session(session_id)

    return {
        "session_id": session_id,
        "occupation": {
            "isco_code": isco_code,
            "title": occupation_title,
            "confidence": classification["confidence"],
            "matched_on": classification["matched_on"]
        },
        "assessment": {
            # total_questions = 2 questions × number of dimensions
            "total_dimensions":  len(dimensions_to_assess),
            "total_questions":   len(dimensions_to_assess) * 2,
            "region": config["region_name"],
            "location": config["location_context"]
        },
        "challenge": challenge_bundle["client"],
        "progress": session.progress()
    }


@app.post("/assess/answer")
def submit_answer(req: AnswerRequest):
    """
    Step 2+: Submit answer, get evaluation and next challenge (or completion).

    2-question-per-dimension flow:
      Q1 → store score → generate Q2 for the same dimension
      Q2 → average Q1+Q2 scores → advance dimension → generate Q1 for next dimension

    Runs Pipeline 4b (eval) → P5 (advance on Q2) → P4 (next challenge) or P6 (profile)
    """
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.completed:
        raise HTTPException(status_code=400, detail="Assessment already completed")

    challenge = session._current_challenge
    config    = session._config

    # P4b: Score MCQ answer — deterministic, no API call
    evaluation = score_mcq_answer(
        challenge=challenge,
        selected_id=req.selected_option.upper()
    )

    current_dim_id = session.current_dimension()
    q_num = getattr(session, "_current_q_num", 1)

    if q_num == 1:
        # ── First question for this dimension ────────────────────────────────
        # Store the raw score; do NOT advance the dimension yet.
        if not hasattr(session, "_first_scores") or session._first_scores is None:
            session._first_scores = {}
        session._first_scores[current_dim_id] = evaluation["score"]
        session._current_q_num = 2
        persist_session(session.session_id)

        # Generate a second, distinct question for the same dimension
        current_dimension = session._dimension_plan[current_dim_id]
        q2_bundle = generate_challenge(
            occupation_title=session.occupation_title,
            dimension=current_dimension,
            country_config=config,
            tier=session.current_tier
        )
        session._current_challenge = q2_bundle["full"]
        persist_session(session.session_id)

        progress = session.progress()
        return {
            "evaluation":         evaluation,
            "progress":           progress,
            "assessment_complete": False,
            "next_challenge":     q2_bundle["client"],
            "question_number":    2,          # tells the UI this is Q2 of 2 for same dimension
            "questions_per_dim":  2
        }

    else:
        # ── Second question for this dimension ───────────────────────────────
        # Average Q1 and Q2 scores to get the dimension score.
        first_scores = getattr(session, "_first_scores", {})
        q1_score = first_scores.get(current_dim_id, evaluation["score"])
        avg_score = (q1_score + evaluation["score"]) // 2

        # Rebuild evaluation with averaged score and recalculated tier/pass
        current_tier = session.current_tier
        if avg_score >= 70:
            tier_achieved = current_tier
            passed        = True
        elif avg_score >= 35:
            tier_achieved = max(1, current_tier - 1)
            passed        = False
        else:
            tier_achieved = 1
            passed        = False

        averaged_eval = {
            **evaluation,                    # keeps feedback, signals from Q2
            "score":          avg_score,
            "tier_achieved":  tier_achieved,
            "passed":         passed,
        }

        # Reset to Q1 for the next dimension
        session._current_q_num = 1
        if current_dim_id in session._first_scores:
            del session._first_scores[current_dim_id]

        # P5: Advance dimension with averaged result
        record_answer(req.session_id, averaged_eval)
        progress = session.progress()

        if session.completed:
            return {
                "evaluation":         averaged_eval,
                "progress":           progress,
                "assessment_complete": True,
                "next_challenge":     None,
                "question_number":    2,
                "questions_per_dim":  2,
                "message":            "Assessment complete. Generating your skill profile..."
            }

        # P4: Generate Q1 for the next dimension
        next_dim_id   = session.current_dimension()
        next_dimension = session._dimension_plan[next_dim_id]

        next_bundle = generate_challenge(
            occupation_title=session.occupation_title,
            dimension=next_dimension,
            country_config=config,
            tier=session.current_tier
        )
        session._current_challenge = next_bundle["full"]
        persist_session(session.session_id)

        return {
            "evaluation":         averaged_eval,
            "progress":           progress,
            "assessment_complete": False,
            "next_challenge":     next_bundle["client"],
            "question_number":    1,          # Q1 of 2 for the next dimension
            "questions_per_dim":  2
        }


@app.get("/assess/profile/{session_id}")
def get_profile(session_id: str):
    """
    Final step: Generate and return the full skill profile.
    Runs Pipeline 6 (localise + profile generation)
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.completed:
        raise HTTPException(status_code=400, detail="Assessment not yet complete")

    config = session._config

    # P6: Generate profile
    profile = generate_profile(
        occupation_title=session.occupation_title,
        dimension_results=session.dimension_results,
        country_config=config,
        education_level=getattr(session, '_education_level', 'upper_secondary'),
        experience_years=getattr(session, '_experience_years', 0),
        other_skills=getattr(session, '_other_skills', ''),
        isco_code=session.isco_code
    )

    # Store profile ID on session so worker registry can surface it
    if profile.get("profile_id"):
        session._profile_id = profile["profile_id"]
        persist_session(session_id)   # flush profile_id to SQLite

    # Persist profile permanently for employer verification lookup
    from pipelines.scoring import compute_weighted_score
    overall_score = compute_weighted_score(session.dimension_results, session.isco_code)
    if profile.get("profile_id"):
        save_profile(
            profile_id      = profile["profile_id"],
            session_id      = session_id,
            region_id       = session.region_id,
            occupation_title= session.occupation_title,
            isco_code       = session.isco_code,
            overall_score   = overall_score,
            profile_data    = profile
        )

    return {
        "session_id": session_id,
        "profile": profile
    }


@app.get("/opportunities/{session_id}")
def get_opportunities(session_id: str):
    """
    Module 3 — Amara's view: match skill profile to local opportunities.
    Runs Pipeline 7 (opportunity matcher)
    Returns 3 layers: ready_now, close_gap, training_pathway
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.completed:
        raise HTTPException(status_code=400, detail="Assessment not yet complete")

    config = session._config

    opportunities = match_opportunities(
        dimension_results=session.dimension_results,
        isco_code=session.isco_code,
        region_id=session.region_id,
        occupation_title=session.occupation_title,
        education=getattr(session, '_education_level', ''),
        experience_years=getattr(session, '_experience_years', 0)
    )

    return {
        "session_id": session_id,
        "occupation_title": session.occupation_title,
        "region": config["region_name"],
        "currency": config["currency"],
        "opportunities": opportunities
    }


@app.get("/dashboard/{region}")
def get_dashboard(region: str):
    """
    Module 3 — Policymaker view: aggregate skill and opportunity signals.
    Returns skill gap heatmap, sector demand, opportunity distribution, country comparison.
    """
    import json
    from pathlib import Path as P

    config = load_config(region)

    # Load opportunity seed data for econometric signals
    seed_dir = BASE_DIR / "data" / "seed"
    country = region.split("_")[0]
    opp_file = seed_dir / f"opportunities_{country}.json"

    opportunities_data = {}
    if opp_file.exists():
        with open(opp_file) as f:
            opportunities_data = json.load(f)

    opportunities = opportunities_data.get("opportunities", [])
    meta = opportunities_data.get("meta", {})

    # Aggregate from live sessions (anonymised)
    from pipelines.p5_difficulty import _sessions
    region_sessions = [s for s in _sessions.values() if s.region_id == region and s.completed]

    # Skill gap aggregate
    dimension_aggregates = {}
    for session in region_sessions:
        for dim_id, result in session.dimension_results.items():
            if dim_id not in dimension_aggregates:
                dimension_aggregates[dim_id] = {
                    "label": result["dimension_label"],
                    "scores": [],
                    "tiers": []
                }
            dimension_aggregates[dim_id]["scores"].append(result["score"])
            dimension_aggregates[dim_id]["tiers"].append(result["tier_achieved"])

    skill_gaps = []
    for dim_id, agg in dimension_aggregates.items():
        scores = agg["scores"]
        avg_score = int(sum(scores) / len(scores)) if scores else 0
        avg_tier = round(sum(agg["tiers"]) / len(agg["tiers"]), 1) if agg["tiers"] else 1.0
        skill_gaps.append({
            "dimension": dim_id,
            "label": agg["label"],
            "avg_score": avg_score,
            "avg_tier": avg_tier,
            "assessed_workers": len(scores)
        })

    # Sector demand signals from seed data
    sector_signals = []
    for opp in opportunities:
        sig = opp.get("econometric_signals", {})
        if opp.get("type") != "training_pathway":
            sector_signals.append({
                "sector": sig.get("sector_label", ""),
                "employer_demand_pct": sig.get("employer_demand_pct", 0),
                "sector_growth_pct": sig.get("sector_growth_pct", 0),
                "wage_floor": sig.get("wage_floor_monthly", 0),
                "currency": meta.get("currency", ""),
                "opportunity_type": opp.get("type_label", "")
            })

    # Opportunity type distribution from seed
    type_distribution = {}
    for opp in opportunities:
        t = opp.get("type_label", "Other")
        type_distribution[t] = type_distribution.get(t, 0) + 1

    return {
        "region": config["region_name"],
        "country": config["country"],
        "total_assessed_workers": len(region_sessions),
        "skill_gaps": skill_gaps,
        "sector_signals": sector_signals,
        "opportunity_type_distribution": type_distribution,
        "econometric_sources": meta.get("econometric_sources", {}),
        "isco_priority_groups": config.get("isco_priority_groups", [])
    }


@app.get("/workers/{region}")
def get_worker_registry(
    region: str,
    dimension: Optional[str] = Query(None, description="Filter by dimension ID, e.g. fault_diagnosis"),
    min_tier: Optional[int] = Query(None, ge=1, le=3, description="Minimum tier in the filtered dimension (1=Entry, 2=Functional, 3=Mastery)"),
    min_score: Optional[int] = Query(0, ge=0, le=100, description="Minimum overall score"),
    isco_group: Optional[int] = Query(None, ge=1, le=9, description="Filter by ISCO-08 major group (1-9)")
):
    """
    Policymaker worker registry — all completed assessments in a region,
    ranked by overall score, with dimension-level filtering.

    Filter examples:
      /workers/ghana_urban?dimension=fault_diagnosis&min_tier=2
      /workers/ghana_urban?min_score=60&isco_group=7
    """
    config = load_config(region)

    from pipelines.p5_difficulty import _sessions
    from pipelines.scoring import compute_weighted_score

    # All completed sessions for this region
    region_sessions = [
        s for s in _sessions.values()
        if s.region_id == region and s.completed
    ]

    TIER_LABELS = {0: "—", 1: "Entry", 2: "Functional", 3: "Mastery"}
    TIER_COLORS = {0: "#334155", 1: "#64748b", 2: "#3b82f6", 3: "#22c55e"}

    workers = []
    for session in region_sessions:
        overall_score = compute_weighted_score(session.dimension_results, session.isco_code)

        # Build dimension snapshot
        dims = {}
        for dim_id, result in session.dimension_results.items():
            dims[dim_id] = {
                "score": result.get("score", 0),
                "tier": result.get("tier_achieved", 0),
                "tier_label": TIER_LABELS.get(result.get("tier_achieved", 0), "—"),
                "tier_color": TIER_COLORS.get(result.get("tier_achieved", 0), "#334155"),
                "label": result.get("dimension_label", dim_id)
            }

        # Pull profile ID if profile was generated
        profile_id = getattr(session, "_profile_id", None)

        workers.append({
            "session_id": session.session_id,
            "session_id_short": session.session_id[:8].upper(),
            "occupation_title": session.occupation_title,
            "isco_code": session.isco_code,
            "isco_group": int(session.isco_code[0]) if session.isco_code else None,
            "overall_score": overall_score,
            "education": getattr(session, "_education_level", "—"),
            "experience_years": getattr(session, "_experience_years", 0),
            "dimensions": dims,
            "profile_id": profile_id
        })

    # ── Filters ──────────────────────────────────────────────────────────────

    # Minimum overall score
    if min_score:
        workers = [w for w in workers if w["overall_score"] >= min_score]

    # ISCO group
    if isco_group is not None:
        workers = [w for w in workers if w["isco_group"] == isco_group]

    # Dimension + tier filter
    if dimension:
        workers = [
            w for w in workers
            if dimension in w["dimensions"] and
               w["dimensions"][dimension]["tier"] >= (min_tier or 1)
        ]

    # ── Sort by overall score descending, assign rank ─────────────────────────
    workers.sort(key=lambda w: w["overall_score"], reverse=True)
    for i, w in enumerate(workers):
        w["rank"] = i + 1

    # Dimension index (all dimensions seen across all workers — for filter UI)
    all_dims = {}
    for s in _sessions.values():
        if s.region_id == region and s.completed:
            for dim_id, result in s.dimension_results.items():
                if dim_id not in all_dims:
                    all_dims[dim_id] = result.get("dimension_label", dim_id)

    return {
        "region": config["region_name"],
        "country": config["country"],
        "total_assessed": len([s for s in _sessions.values() if s.region_id == region and s.completed]),
        "filtered_count": len(workers),
        "active_filters": {
            "dimension": dimension,
            "min_tier": min_tier,
            "min_score": min_score,
            "isco_group": isco_group
        },
        "available_dimensions": all_dims,
        "workers": workers
    }


@app.get("/verify/{profile_id}")
def verify_profile(profile_id: str):
    """
    Public employer verification endpoint.
    Returns a sanitised skill card — no PII, no session data.
    An employer pastes the profile ID and gets back verifiable facts.
    """
    record = db_get_profile(profile_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Profile '{profile_id}' not found. The ID may be incorrect or the assessment was not completed."
        )

    profile_data = record["data"]

    # Sanitised response — only what an employer needs to see
    TIER_LABELS = {0: "Not assessed", 1: "Entry", 2: "Functional", 3: "Mastery"}
    TIER_COLORS = {0: "#334155",      1: "#64748b", 2: "#3b82f6",   3: "#22c55e"}

    dimensions = []
    for dim in profile_data.get("dimension_summary", []):
        dimensions.append({
            "id":         dim.get("id", ""),
            "label":      dim.get("label", ""),
            "score":      dim.get("score", 0),
            "tier":       dim.get("tier", 0),
            "tier_label": dim.get("tier_label", TIER_LABELS.get(dim.get("tier", 0))),
            "tier_color": TIER_COLORS.get(dim.get("tier", 0), "#334155")
        })

    return {
        "verified":         True,
        "profile_id":       profile_id,
        "issued_by":        "UNMAPPED Skills Verification System",
        "taxonomy":         "ISCO-08 · O*NET 28.0 · ESCO v1.2.1",
        "verified_at":      record["created_at"],
        "occupation_title": record["occupation_title"],
        "isco_code":        record["isco_code"],
        "region":           profile_data.get("region", record["region_id"]),
        "overall_score":    record["overall_score"],
        "employer_signal":  profile_data.get("employer_signal", ""),
        "dimensions":       dimensions,
        # Portability metadata
        "transferable_to":  profile_data.get("skills_card", {})
                                        .get("portability", {})
                                        .get("transferable_to", [])
    }


@app.get("/card/{profile_id}")
def serve_card(profile_id: str):
    """Serve the shareable verification card page for a given profile ID."""
    if not profile_exists(profile_id):
        raise HTTPException(status_code=404, detail="Profile not found.")
    return FileResponse(UI_DIR / "verify.html")


# Serve the UI
@app.get("/")
def serve_ui():
    return FileResponse(UI_DIR / "index.html")


@app.get("/dashboard")
def serve_dashboard():
    return FileResponse(UI_DIR / "dashboard.html")
