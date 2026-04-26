# HACK-NATION-2026

UNMAPPED is a plug-and-play infrastructure layer that converts informal skills into verifiable, portable economic signals and connects them to real, local opportunities using grounded economic data.

## Backend Prototype

This repository contains a configurable FastAPI backend with:

- `Skills Signal Engine`
- `Evidence / Trust Engine`
- `AI Risk Engine`
- `Opportunity Matching Engine`
- `Dashboard Engine`
- `Config Loader`

## Architecture Flow

`Country Config -> User Input -> Skill Mapping -> Trust Scoring -> AI Risk -> Opportunity Matching -> Readiness Explanation -> Next Steps -> Dashboard`

## Data Signals Included

- `Wage signal` from ILOSTAT-style opportunity data
- `Sector growth signal` from WDI-style and labor-market seed data
- `Automation risk signal` from Frey-Osborne-style calibration

## Readiness Score

`Score = Skill Match (40%) + Evidence (20%) + Local Demand (15%) + Learning Gap (15%) + AI Resilience (10%)`

Every returned opportunity includes a short breakdown showing why the score exists.

## Country-Agnostic Configs

Two sample country configurations are included:

- `config/ghana_urban.yaml`
- `config/kenya_mixed.yaml`

Change taxonomy, language register, opportunity types, and AI risk calibration in config only.

Each config includes `country_code` so signal datasets can be mapped without code changes.

The repo supports a text-first intake flow plus an optional short 2-3 minute work video for sectors like agriculture, electrician work, mechanics, repairs, and similar hands-on roles. The video is treated as evidence for the same skill pipeline, not a separate product path.

For users who do not want to type in English, the browser UI also supports voice input. That lets a user speak their work description in a supported browser language, then route the transcription into the same assessment flow.

## Live Research Layer

Tavily is wired as an optional discovery layer for external labor-market research and opportunity refreshes.

- `GET /research/opportunities`
- `GET /research/context`

The research layer is discovery only. The local scoring engines still decide readiness, risk, and match quality.

## Country Education Systems

The `Highest education level` dropdown is populated from the selected country config, so the labels can reflect local education systems instead of one global list.

## One-Page CV

Before matching local opportunities, the system generates a one-page, human-readable skills card that acts like a CV:

- occupation and ISCO mapping
- assessed strengths
- growth areas
- evidence-backed employer signal
- portable profile ID for verification

## How Skills Are Verified and Assessed

Informal skills are verified through short task-based assessment, not self-claim alone.

- user description maps to a standard occupation
- prompts or tasks test the real work dimensions for that occupation
- answers are scored by dimension
- the system turns the result into a portable skills profile
- employers can verify the profile ID on a public endpoint

## How Job Matching Works

The system does not suggest fantasy careers. It returns:

- `ready_now`
- `close_gap`
- `training_pathway`

Each match is explained in plain language, including what the worker already does well, what is missing, and how to close the gap.

## Run

```powershell
& ".venv\Scripts\Activate.ps1"
pip install -r requirements.txt
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open the app at `http://127.0.0.1:8000/`

API docs are at `http://127.0.0.1:8000/docs`

## Useful Endpoints

- `GET /regions`
- `POST /assess/start`
- `POST /assess/answer`
- `GET /assess/profile/{session_id}`
- `POST /assess/evidence/video`
- `GET /assess/risk/{session_id}`
- `GET /opportunities/{session_id}`
- `GET /dashboard/{region}`
- `GET /workers/{region}`
- `GET /verify/{profile_id}`
- `GET /config-diff`

## Example Request

```json
{
  "description": "I repair mobile phones and help customers choose accessories",
  "region": "ghana_urban",
  "education_level": "upper_secondary",
  "experience_years": 5,
  "other_skills": "customer service, basic coding"
}
```

## Repo Layout

- `api/` FastAPI routes and app wiring
- `app/` shared config and services
- `config/` country-localized YAML files
- `data/` seed data and local persistence
- `database/` SQLite storage helpers
- `pipelines/` assessment, profile, opportunity, and risk logic
- `ui/` browser-facing worker and policymaker interfaces
- `scripts/` maintenance and preprocessing helpers
