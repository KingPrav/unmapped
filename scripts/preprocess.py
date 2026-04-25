"""
UNMAPPED — Data Preprocessing Script
Run this once after downloading raw data files to build the SQLite occupation index.

Usage:
  python scripts/preprocess.py

Expected raw files:
  data/raw/isco08/         → ISCO-08 structure CSV from ILO
  data/raw/onet/           → O*NET Task Statements, DWA Reference, Tasks to DWAs, Occupation Data
  data/raw/crosswalk/      → ISCO-08_Crosswalk.xlsx from O*NET Center
  data/raw/esco/           → occupations_en.csv, skills_en.csv (+ _fr versions)
  data/raw/ilostat/        → ghana_employment.csv, kenya_employment.csv
"""

import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RAW = BASE_DIR / "data" / "raw"
DB_PATH = BASE_DIR / "data" / "processed" / "occupation_index.db"
DIM_MAP_PATH = BASE_DIR / "data" / "seed" / "dwa_dimension_map.json"

import json
with open(DIM_MAP_PATH) as f:
    DIMENSION_MAP = json.load(f)["dimensions"]


def keyword_to_dimension(dwa_text: str) -> str:
    text_lower = dwa_text.lower()
    scores = {}
    for dim_id, dim_data in DIMENSION_MAP.items():
        score = sum(1 for kw in dim_data["keywords"] if kw in text_lower)
        scores[dim_id] = score
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "fault_diagnosis"


def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS isco_occupations (
            isco_code TEXT PRIMARY KEY,
            isco_group INTEGER,
            title TEXT,
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS onet_occupations (
            onet_code TEXT PRIMARY KEY,
            title TEXT,
            isco_code TEXT
        );

        CREATE TABLE IF NOT EXISTS dwas (
            id TEXT,
            onet_code TEXT,
            isco_code TEXT,
            dwa_text TEXT,
            dimension TEXT,
            PRIMARY KEY (id, onet_code)
        );

        CREATE TABLE IF NOT EXISTS esco_labels (
            isco_code TEXT,
            language TEXT,
            label TEXT,
            PRIMARY KEY (isco_code, language)
        );

        CREATE TABLE IF NOT EXISTS ilostat_employment (
            country_code TEXT,
            isco_group INTEGER,
            isco_title TEXT,
            employment_share REAL,
            year INTEGER,
            PRIMARY KEY (country_code, isco_group)
        );
    """)
    conn.commit()
    print("✓ Database schema created")


def load_isco08(conn):
    isco_dir = RAW / "isco08"
    csv_files = list(isco_dir.glob("*.csv"))
    if not csv_files:
        print("⚠ No ISCO-08 CSV found in data/raw/isco08/ — skipping")
        return

    df = pd.read_csv(csv_files[0], dtype=str)
    # Normalise column names (ILO exports vary)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Try to find code and title columns
    code_col = next((c for c in df.columns if "code" in c), None)
    title_col = next((c for c in df.columns if "title" in c or "name" in c), None)

    if not code_col or not title_col:
        print(f"⚠ Could not find code/title columns in ISCO-08 file. Columns: {list(df.columns)}")
        return

    for _, row in df.iterrows():
        code = str(row[code_col]).strip()
        if len(code) == 4:  # Unit group level only
            conn.execute(
                "INSERT OR REPLACE INTO isco_occupations (isco_code, isco_group, title) VALUES (?, ?, ?)",
                (code, int(code[0]), str(row[title_col]).strip())
            )
    conn.commit()
    print(f"✓ ISCO-08 loaded")


def load_onet_crosswalk(conn):
    cw_dir = RAW / "crosswalk"
    xlsx_files = list(cw_dir.glob("*.xlsx"))
    if not xlsx_files:
        print("⚠ No crosswalk XLSX found in data/raw/crosswalk/ — skipping")
        return

    df = pd.read_excel(xlsx_files[0], dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    onet_col = next((c for c in df.columns if "o*net" in c or "onet" in c or "soc" in c), None)
    isco_col = next((c for c in df.columns if "isco" in c), None)

    if not onet_col or not isco_col:
        print(f"⚠ Could not find O*NET/ISCO columns in crosswalk. Columns: {list(df.columns)}")
        return

    for _, row in df.iterrows():
        onet_code = str(row[onet_col]).strip()
        isco_code = str(row[isco_col]).strip()
        if onet_code and isco_code and len(isco_code) == 4:
            conn.execute(
                "INSERT OR REPLACE INTO onet_occupations (onet_code, isco_code) VALUES (?, ?)",
                (onet_code, isco_code)
            )
    conn.commit()
    print("✓ ISCO→O*NET crosswalk loaded")


def load_onet_dwas(conn):
    onet_dir = RAW / "onet"

    task_file = next(onet_dir.glob("Task Statements*"), None)
    dwa_ref_file = next(onet_dir.glob("DWA Reference*"), None)
    task_dwa_file = next(onet_dir.glob("Tasks to DWAs*"), None)

    if not all([task_file, dwa_ref_file, task_dwa_file]):
        print("⚠ O*NET files missing in data/raw/onet/ — skipping DWA load")
        return

    # Load DWA reference (id → text)
    dwa_ref = pd.read_csv(dwa_ref_file, sep="\t", dtype=str)
    dwa_ref.columns = [c.strip().lower().replace(" ", "_") for c in dwa_ref.columns]
    dwa_id_col = next((c for c in dwa_ref.columns if "dwa" in c and "id" in c), dwa_ref.columns[0])
    dwa_text_col = next((c for c in dwa_ref.columns if "title" in c or "text" in c or "description" in c), dwa_ref.columns[1])
    dwa_lookup = dict(zip(dwa_ref[dwa_id_col], dwa_ref[dwa_text_col]))

    # Load tasks→DWAs mapping
    task_dwa = pd.read_csv(task_dwa_file, sep="\t", dtype=str)
    task_dwa.columns = [c.strip().lower().replace(" ", "_") for c in task_dwa.columns]

    onet_col = next((c for c in task_dwa.columns if "o*net" in c or "onet" in c or "soc" in c), None)
    dwa_col = next((c for c in task_dwa.columns if "dwa" in c and "id" in c), None)

    if not onet_col or not dwa_col:
        print(f"⚠ Cannot find O*NET/DWA columns in Tasks to DWAs file. Columns: {list(task_dwa.columns)}")
        return

    count = 0
    for _, row in task_dwa.iterrows():
        onet_code = str(row[onet_col]).strip()
        dwa_id = str(row[dwa_col]).strip()
        dwa_text = dwa_lookup.get(dwa_id, "")

        if not dwa_text:
            continue

        # Get ISCO code from crosswalk
        cursor = conn.execute("SELECT isco_code FROM onet_occupations WHERE onet_code = ?", (onet_code,))
        row_result = cursor.fetchone()
        if not row_result:
            continue

        isco_code = row_result[0]
        dimension = keyword_to_dimension(dwa_text)

        conn.execute(
            "INSERT OR REPLACE INTO dwas (id, onet_code, isco_code, dwa_text, dimension) VALUES (?, ?, ?, ?, ?)",
            (dwa_id, onet_code, isco_code, dwa_text, dimension)
        )
        count += 1

    conn.commit()
    print(f"✓ O*NET DWAs loaded: {count} entries")


def load_esco(conn):
    esco_dir = RAW / "esco"
    for lang in ["en", "fr"]:
        occ_file = next(esco_dir.glob(f"occupations_{lang}*"), None)
        if not occ_file:
            print(f"⚠ ESCO occupations_{lang} not found — skipping")
            continue

        df = pd.read_csv(occ_file, dtype=str)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        code_col = next((c for c in df.columns if "isco" in c or "code" in c), None)
        label_col = next((c for c in df.columns if "label" in c or "title" in c or "preferred" in c), None)

        if not code_col or not label_col:
            print(f"⚠ Cannot find code/label columns in ESCO {lang} file. Columns: {list(df.columns)}")
            continue

        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            label = str(row[label_col]).strip()
            if code and label and len(code) >= 4:
                isco_code = code[:4]
                conn.execute(
                    "INSERT OR REPLACE INTO esco_labels (isco_code, language, label) VALUES (?, ?, ?)",
                    (isco_code, lang, label)
                )
        conn.commit()
        print(f"✓ ESCO {lang.upper()} labels loaded")


def load_ilostat(conn):
    ilostat_dir = RAW / "ilostat"
    for country, code in [("ghana", "GHA"), ("kenya", "KEN")]:
        csv_file = next(ilostat_dir.glob(f"{country}*.csv"), None)
        if not csv_file:
            print(f"⚠ ILOSTAT file for {country} not found — skipping")
            continue

        df = pd.read_csv(csv_file, dtype=str)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        isco_col = next((c for c in df.columns if "isco" in c or "occupation" in c), None)
        share_col = next((c for c in df.columns if "share" in c or "percent" in c or "value" in c), None)
        year_col = next((c for c in df.columns if "year" in c or "time" in c), None)

        if not isco_col:
            print(f"⚠ Cannot find ISCO column in ILOSTAT {country} file. Columns: {list(df.columns)}")
            continue

        for _, row in df.iterrows():
            try:
                isco_group = int(str(row[isco_col]).strip()[0])
                share = float(row[share_col]) if share_col else 0.0
                year = int(row[year_col]) if year_col else 2022
                conn.execute(
                    "INSERT OR REPLACE INTO ilostat_employment VALUES (?, ?, ?, ?, ?)",
                    (code, isco_group, str(row[isco_col]).strip(), share, year)
                )
            except Exception:
                continue

        conn.commit()
        print(f"✓ ILOSTAT {country.title()} loaded")


def main():
    print("\n🔧 UNMAPPED — Data Preprocessing\n" + "─" * 40)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    load_isco08(conn)
    load_onet_crosswalk(conn)
    load_onet_dwas(conn)
    load_esco(conn)
    load_ilostat(conn)

    conn.close()
    print("\n✅ Preprocessing complete. Database saved to:", DB_PATH)
    print("   The API will now use real data instead of seed data.\n")


if __name__ == "__main__":
    main()
