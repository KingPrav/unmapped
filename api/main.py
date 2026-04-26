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
  GET  /health                          — Health check
"""

import os
import uuid
import yaml
from pathlib import Path
from fastapi import FastAPI, HTTPException
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
from pipelines.p4_challenge_generator import generate_challenge, evaluate_answer
from pipelines.p5_difficulty import create_session, get_session, record_answer
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
    answer: str


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
    session._dimension_plan = dimension_result["dimension_plan"]
    session._config = config
    session._education_level = req.education_level
    session._experience_years = req.experience_years
    session._other_skills = req.other_skills

    # P4: Generate first challenge
    first_dim_id = session.current_dimension()
    first_dimension = dimension_result["dimension_plan"][first_dim_id]

    challenge = generate_challenge(
        occupation_title=occupation_title,
        dimension=first_dimension,
        country_config=config,
        tier=session.current_tier
    )

    session._current_challenge = challenge

    return {
        "session_id": session_id,
        "occupation": {
            "isco_code": isco_code,
            "title": occupation_title,
            "confidence": classification["confidence"],
            "matched_on": classification["matched_on"]
        },
        "assessment": {
            "total_dimensions": len(dimensions_to_assess),
            "region": config["region_name"],
            "location": config["location_context"]
        },
        "challenge": challenge,
        "progress": session.progress()
    }


@app.post("/assess/answer")
def submit_answer(req: AnswerRequest):
    """
    Step 2+: Submit answer, get evaluation and next challenge (or completion).
    Runs Pipeline 4b (eval) → P5 (advance) → P4 (next challenge) or P6 (profile)
    """
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.completed:
        raise HTTPException(status_code=400, detail="Assessment already completed")

    challenge = session._current_challenge
    config = session._config

    # P4b: Evaluate answer
    evaluation = evaluate_answer(
        challenge=challenge,
        user_answer=req.answer,
        occupation_title=session.occupation_title
    )

    # P5: Record result and advance
    record_answer(req.session_id, evaluation)
    progress = session.progress()

    # Check if assessment complete
    if session.completed:
        return {
            "evaluation": evaluation,
            "progress": progress,
            "assessment_complete": True,
            "next_challenge": None,
            "message": "Assessment complete. Generating your skill profile..."
        }

    # P4: Generate next challenge
    next_dim_id = session.current_dimension()
    next_dimension = session._dimension_plan[next_dim_id]

    next_challenge = generate_challenge(
        occupation_title=session.occupation_title,
        dimension=next_dimension,
        country_config=config,
        tier=session.current_tier
    )

    session._current_challenge = next_challenge

    return {
        "evaluation": evaluation,
        "progress": progress,
        "assessment_complete": False,
        "next_challenge": next_challenge
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


# Serve the UI
@app.get("/")
def serve_ui():
    return FileResponse(UI_DIR / "index.html")


@app.get("/dashboard")
def serve_dashboard():
    return FileResponse(UI_DIR / "dashboard.html")
