# UNMAPPED
### Skills Visibility Infrastructure for the Informal Economy
*World Bank Youth Summit — Hackathon Submission*

---

## The Problem

600 million informal economy workers across LMICs have real, demonstrable skills — but no portable proof of them. Without formal credentials, they are invisible to employers, financial services, and policymakers. UNMAPPED changes that.

---

## What We Built

A full-stack skills assessment and verification system that takes a worker from a one-line description of their job to a portable, employer-verifiable skill card — in under 10 minutes, on a phone, with no typing required.

**Three user-facing modules:**

**Module 1 — Worker Assessment (Amara's view)**
The worker describes their occupation in plain language. UNMAPPED classifies it against ISCO-08, retrieves relevant O*NET Detailed Work Activities, maps them to 5 skill dimensions, and generates adaptive scenario-based MCQ challenges. Workers tap one of 4 options on their phone — no typing needed. After 2 rounds per dimension (10 questions total), a portable skill card is generated with a unique Profile ID.

**Module 2 — Opportunity Matching (Amara's view)**
Based on the skill card, UNMAPPED matches the worker to real local opportunities in 3 honest layers — *Ready Now*, *Close Gap*, and *Training Pathway* — grounded in ILOSTAT wage data and World Bank WBES employer demand signals. Matching is brutally honest: low scores don't get aspirational suggestions.

**Module 3 — Policymaker Dashboard**
Aggregate skill gap heatmap, sector demand signals, ISCO priority groups, and a ranked worker registry filterable by skill dimension, tier, score, and ISCO group — with a slide-out skill card drawer per worker.

**Employer Verification**
Every skill card has a unique Profile ID and a shareable URL (`/card/{profile_id}`). Employers open the link and see a verified card with dimension scores, tier badges, and an employer signal — no login, no PII.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI + Python | Clean async REST, auto docs at `/docs` |
| LLM | GPT-4o (OpenAI) | MCQ generation, profile summaries, opportunity explanations |
| Taxonomy | ISCO-08 + O*NET 28.0 + ESCO v1.2.1 | Globally portable, ILO-recognised |
| Database | SQLite (built-in) | Zero-dependency persistence, survives restarts |
| Frontend | Vanilla HTML/CSS/JS | Runs on any phone browser, no framework overhead |
| Config | YAML per region | Swap country context without touching code |

---

## API Endpoints

### Assessment Flow
```
POST /assess/start
     Body: { description, region, education_level, experience_years, other_skills }
     → Classifies occupation, generates first MCQ challenge

POST /assess/answer
     Body: { session_id, selected_option }
     → Scores answer (deterministic, no API call), returns next challenge
     → 2 questions per dimension, scores averaged before advancing
     → Returns question_number (1 or 2) so UI shows Q1 OF 2 / Q2 OF 2

GET  /assess/profile/{session_id}
     → Generates portable skill card with ISCO-weighted overall score
```

### Opportunity Matching
```
GET  /opportunities/{session_id}
     → Returns 3 layers: ready_now, close_gap, training_pathway
     → Grounded in ILOSTAT + WBES econometric signals
     → ISCO-aware matching: exact group required for ready_now
```

### Policymaker
```
GET  /dashboard/{region}
     → Aggregate skill gaps, sector demand, ISCO priority groups
     → Powered by live session data + seed opportunity data

GET  /workers/{region}?dimension=&min_tier=&min_score=&isco_group=
     → Ranked worker registry (all completed assessments in region)
     → Filterable by dimension, tier, score, ISCO major group
```

### Employer Verification
```
GET  /verify/{profile_id}
     → Public endpoint — sanitised skill card (no PII)
     → Returns: occupation, ISCO, score, dimension tiers, employer signal

GET  /card/{profile_id}
     → Serves shareable employer-facing verification page (verify.html)
```

### Utility
```
GET  /regions       → Lists all available region configs with language info
GET  /health        → Health check
GET  /              → Worker assessment UI (index.html)
GET  /dashboard     → Policymaker dashboard (dashboard.html)
```

---

## Pipeline Architecture

```
User input
    │
    ▼
P1 — Occupation Classifier     (GPT-4o) → ISCO-08 code + title
    │
    ▼
P2 — DWA Retriever             (O*NET seed data) → Detailed Work Activities
    │
    ▼
P3 — Dimension Mapper          (keyword matching) → 5 skill dimensions
    │
    ▼
P4 — MCQ Challenge Generator   (GPT-4o) → Scenario questions in worker's language
    │
    ▼
P4b — MCQ Scorer               (deterministic) → 0/50/100, no API call
    │
    ▼
P5 — Adaptive Difficulty       (SQLite) → Tier 1→2→3 based on performance
    │
    ▼
P6 — Profile Generator         (GPT-4o + ISCO weights) → Portable skill card
    │
    ▼
P7 — Opportunity Matcher       (GPT-4o-mini + ILOSTAT/WBES) → 3-layer match
```

---

## Skill Dimensions

| ID | Label | Description |
|---|---|---|
| `fault_diagnosis` | Fault Diagnosis | Identify and fix what's wrong |
| `communication` | Communication | Interact with customers, peers, employers |
| `resource_judgment` | Resource Judgment | Manage materials, money, time |
| `process_quality` | Process Quality | Follow steps, maintain standards |
| `operational_organization` | Operational Organization | Plan, schedule, organise work |

---

## ISCO-Aware Scoring

Overall score is not a flat average — dimensions are weighted by ISCO major group:

| ISCO Group | Heaviest Dimension | Weight |
|---|---|---|
| 7 — Craft & Trade (phone repair, mechanics) | Fault Diagnosis | 40% |
| 5 — Service & Sales (market traders) | Communication | 40% |
| 6 — Agriculture | Resource Judgment | 35% |
| 3 — Technicians | Fault Diagnosis | 35% |
| 4 — Clerical | Communication | 35% |

A phone repair tech with 100/100 on diagnostics and 20/100 on communication scores **68** (weighted) vs **54** (flat). The score reflects what actually matters for that occupation.

---

## Region Configs

| Region ID | Country | Language |
|---|---|---|
| `ghana_urban` | Ghana | English |
| `ghana_twi` | Ghana | Twi (Akan) |
| `kenya_mixed` | Kenya | English |
| `kenya_swahili` | Kenya | Swahili |

Questions, profile summaries, and employer signals are all generated in the worker's chosen language. New regions require only a YAML config file — no code changes.

---

## Scoring Honesty Rules

- **Score floor**: dimension score < 35 counts as one tier below actual (passing a question at 8/100 is not Entry competence)
- **Overall floor**: score < 30 → training pathway only; 30–49 → close gap at best; 50+ → normal matching
- **ISCO exact match** required for `ready_now`; adjacent groups only for `close_gap`
- **Experience credit**: 5+ years field experience or vocational/tertiary education can lift a marginal `training_pathway` to `close_gap` (never to `ready_now`)

---

## Running Locally

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set your OpenAI key
echo "OPENAI_API_KEY=sk-proj-..." > .env

# 3. Start the server
uvicorn api.main:app --reload --port 8000

# 4. Open the worker UI
open http://localhost:8000

# 5. Open the policymaker dashboard
open http://localhost:8000/dashboard

# 6. API docs (auto-generated)
open http://localhost:8000/docs
```

---

## Data Sources

| Source | Used For |
|---|---|
| ILOSTAT — Ghana/Kenya Employment by Occupation (ISCO-08), 2022 | Wage floors, sector employment share |
| World Bank Enterprise Survey (WBES) 2022 | Employer demand % by sector |
| O*NET 28.0 (via ISCO-08 crosswalk) | Detailed Work Activities per occupation |
| ESCO v1.2.1 | Skill taxonomy labels, portability metadata |
| ILO ISCO-08 | Occupational classification backbone |

---

## What's Portable

Every skill card is:
- **ISCO-coded** — the same occupational taxonomy used by ILO, World Bank, and EU labour ministries
- **Score-referenced** — 0–100 with explicit tier labels (Entry / Functional / Mastery)
- **Verifiable** — employers open `/card/{profile_id}` and see the same data, no login required
- **Transferable** — recognised for employer verification, training enrollment, financial services, and cross-border employment

---

*Built for the World Bank Youth Summit Hackathon.*
*Infrastructure for the 600 million workers the formal economy can't see.*
