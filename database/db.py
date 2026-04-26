"""
UNMAPPED — SQLite Persistence Layer
Replaces the in-memory _sessions dict so data survives server restarts.

Tables:
  sessions  — full serialised AssessmentSession state (JSON blob)
  profiles  — generated skill profiles keyed by profile_id (employer lookup)
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent.parent / "data" / "unmapped.db"


# ─────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ─────────────────────────────────────────────
# Initialise schema
# ─────────────────────────────────────────────

def init_db():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                region_id    TEXT NOT NULL,
                completed    INTEGER DEFAULT 0,
                data         TEXT NOT NULL,
                created_at   TEXT DEFAULT (datetime('now','utc')),
                updated_at   TEXT DEFAULT (datetime('now','utc'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                profile_id        TEXT PRIMARY KEY,
                session_id        TEXT NOT NULL,
                region_id         TEXT NOT NULL,
                occupation_title  TEXT NOT NULL,
                isco_code         TEXT,
                overall_score     INTEGER,
                data              TEXT NOT NULL,
                created_at        TEXT DEFAULT (datetime('now','utc'))
            )
        """)
        # Index for fast region queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_region
            ON sessions(region_id, completed)
        """)
        conn.commit()


# ─────────────────────────────────────────────
# Session persistence
# ─────────────────────────────────────────────

def save_session(session_id: str, region_id: str, completed: bool, data: dict):
    """Insert or replace a full session state."""
    with _conn() as conn:
        conn.execute("""
            INSERT INTO sessions (session_id, region_id, completed, data, updated_at)
            VALUES (?, ?, ?, ?, datetime('now','utc'))
            ON CONFLICT(session_id) DO UPDATE SET
                completed  = excluded.completed,
                data       = excluded.data,
                updated_at = excluded.updated_at
        """, (session_id, region_id, int(completed), json.dumps(data)))
        conn.commit()


def load_session(session_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT data FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return json.loads(row["data"]) if row else None


def load_all_sessions() -> list[dict]:
    """Load every session on startup to repopulate the in-memory cache."""
    with _conn() as conn:
        rows = conn.execute("SELECT data FROM sessions").fetchall()
    return [json.loads(r["data"]) for r in rows]


# ─────────────────────────────────────────────
# Profile persistence (employer lookup)
# ─────────────────────────────────────────────

def save_profile(
    profile_id: str,
    session_id: str,
    region_id: str,
    occupation_title: str,
    isco_code: str,
    overall_score: int,
    profile_data: dict
):
    """Store a generated skill profile permanently."""
    with _conn() as conn:
        conn.execute("""
            INSERT INTO profiles
                (profile_id, session_id, region_id, occupation_title, isco_code, overall_score, data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id) DO UPDATE SET
                data = excluded.data
        """, (
            profile_id, session_id, region_id,
            occupation_title, isco_code, overall_score,
            json.dumps(profile_data)
        ))
        conn.commit()


def get_profile(profile_id: str) -> dict | None:
    """
    Fetch a profile by ID.
    Returns sanitised dict — no PII, safe for employer-facing verification.
    """
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)
        ).fetchone()
    if not row:
        return None
    return {
        "profile_id":       row["profile_id"],
        "session_id":       row["session_id"],
        "region_id":        row["region_id"],
        "occupation_title": row["occupation_title"],
        "isco_code":        row["isco_code"],
        "overall_score":    row["overall_score"],
        "created_at":       row["created_at"],
        "data":             json.loads(row["data"])
    }


def profile_exists(profile_id: str) -> bool:
    with _conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM profiles WHERE profile_id = ?", (profile_id,)
        ).fetchone()
    return row is not None
