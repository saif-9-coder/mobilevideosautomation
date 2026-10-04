"""
SQLite storage for the GSMArena -> video pipeline.

Tables
------
brands  : id, name, gsmarena_url, logo_path, scraped_at
series  : id, brand_id, name            -- e.g. "Galaxy S", "iPhone", "Redmi Note"
phones  : id, series_id, name, gsmarena_url, image_url, local_image,
          announced, status, specs_json, scraped_at
videos  : id, phone_id, series_id, title, file_path, thumbnail_path,
          style, status, created_at
"""
import json
import os
import sqlite3
from datetime import datetime

DB_PATH = os.environ.get("MVA_DB", os.path.join(os.path.dirname(__file__), "..", "data", "mva.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS brands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    gsmarena_url TEXT,
    logo_path TEXT,
    scraped_at TEXT
);
CREATE TABLE IF NOT EXISTS series (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    brand_id INTEGER NOT NULL REFERENCES brands(id),
    name TEXT NOT NULL,
    UNIQUE (brand_id, name)
);
CREATE TABLE IF NOT EXISTS phones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id INTEGER REFERENCES series(id),
    brand_id INTEGER NOT NULL REFERENCES brands(id),
    name TEXT NOT NULL,
    gsmarena_url TEXT UNIQUE,
    image_url TEXT,
    local_image TEXT,
    announced TEXT,
    status TEXT,
    specs_json TEXT,
    scraped_at TEXT
);
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phone_id INTEGER REFERENCES phones(id),
    series_id INTEGER REFERENCES series(id),
    title TEXT,
    file_path TEXT,
    thumbnail_path TEXT,
    style TEXT DEFAULT 'spec_showcase',
    status TEXT DEFAULT 'pending',
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS phone_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    phone_id INTEGER NOT NULL REFERENCES phones(id),
    url TEXT,
    local_path TEXT,
    position INTEGER DEFAULT 0,
    UNIQUE (phone_id, url)
);
CREATE INDEX IF NOT EXISTS idx_phone_images ON phone_images(phone_id);
CREATE INDEX IF NOT EXISTS idx_phones_brand ON phones(brand_id);
CREATE INDEX IF NOT EXISTS idx_phones_series ON phones(series_id);
"""


def connect(db_path=DB_PATH):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def now():
    return datetime.utcnow().isoformat(timespec="seconds")


# ---- brands ----
def upsert_brand(con, name, gsmarena_url=None):
    con.execute(
        "INSERT INTO brands (name, gsmarena_url, scraped_at) VALUES (?,?,?) "
        "ON CONFLICT(name) DO UPDATE SET gsmarena_url=excluded.gsmarena_url, scraped_at=excluded.scraped_at",
        (name, gsmarena_url, now()),
    )
    con.commit()
    return con.execute("SELECT id FROM brands WHERE name=?", (name,)).fetchone()["id"]


# ---- series ----
SERIES_PATTERNS = [
    r"(Galaxy S)\d*", r"(Galaxy Z Fold)\d*", r"(Galaxy Z Flip)\d*",
    r"(Galaxy A)\d*", r"(Galaxy M)\d*", r"(Galaxy F)\d*",
    r"(Galaxy Tab S)\d*", r"(Galaxy Tab A)\d*", r"(Galaxy Note)\d*",
    r"(Galaxy Xcover)\d*",
    r"(iPhone)(?: \d+)?", r"(Pixel)(?: \d+)?", r"(Pixel Fold)",
    r"(Redmi Note)(?: \d+)?", r"(Redmi)(?: \d+[A-Z]*)?", r"(Poco [A-Z])(?:\d+)?",
    r"(OnePlus)(?: \d+)?", r"(Nord)(?: \w+)?", r"(Xperia)(?: \d+)?",
    r"(Reno)\d*", r"(Find [A-Z])\d*", r"(Mate)(?: \d+)?", r"(Pura)(?: \d+)?",
    r"(Narzo)(?: \d+[A-Z]*)?",
]

_TRIM = {"ultra", "plus", "pro", "max", "fe", "lite", "mini", "5g", "4g", "s", "e", "neo",
         "prime", "core", "active", "t", "c", "x", "rs", "se", "go"}


def guess_series(phone_name):
    """Guess the series name, e.g. 'Galaxy S24 Ultra' -> 'Galaxy S'."""
    import re

    for pat in SERIES_PATTERNS:
        m = re.search(pat, phone_name, re.I)
        if m:
            return m.group(1).strip().title()
    # generic fallback: drop trailing variant tokens (Ultra, Plus, 5G, digits...)
    words = phone_name.split()
    while words and (re.search(r"\d", words[-1]) or words[-1].lower().strip("+") in _TRIM):
        words.pop()
    core = words[-2:] if len(words) >= 2 else words
    return " ".join(core) if core else phone_name


def get_or_create_series(con, brand_id, series_name):
    con.execute(
        "INSERT OR IGNORE INTO series (brand_id, name) VALUES (?,?)", (brand_id, series_name)
    )
    con.commit()
    return con.execute(
        "SELECT id FROM series WHERE brand_id=? AND name=?", (brand_id, series_name)
    ).fetchone()["id"]


# ---- phones ----
def upsert_phone(con, brand_id, series_id, name, gsmarena_url=None, image_url=None,
                 announced=None, status=None, specs=None):
    con.execute(
        """INSERT INTO phones (series_id, brand_id, name, gsmarena_url, image_url,
                               announced, status, specs_json, scraped_at)
           VALUES (?,?,?,?,?,?,?,?,?)
           ON CONFLICT(gsmarena_url) DO UPDATE SET
             series_id=excluded.series_id, name=excluded.name, image_url=excluded.image_url,
             announced=excluded.announced, status=excluded.status,
             specs_json=excluded.specs_json, scraped_at=excluded.scraped_at""",
        (series_id, brand_id, name, gsmarena_url, image_url, announced, status,
         json.dumps(specs or {}, ensure_ascii=False), now()),
    )
    con.commit()


def get_phone(con, phone_id):
    row = con.execute("SELECT * FROM phones WHERE id=?", (phone_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["specs"] = json.loads(d["specs_json"] or "{}")
    d["images"] = [r["local_path"] for r in con.execute(
        "SELECT local_path FROM phone_images WHERE phone_id=? AND local_path IS NOT NULL "
        "ORDER BY position", (phone_id,)).fetchall()]
    return d


def add_phone_image(con, phone_id, url, local_path, position=0):
    con.execute(
        "INSERT INTO phone_images (phone_id, url, local_path, position) VALUES (?,?,?,?) "
        "ON CONFLICT(phone_id, url) DO UPDATE SET local_path=excluded.local_path, "
        "position=excluded.position",
        (phone_id, url, local_path, position))
    con.commit()
