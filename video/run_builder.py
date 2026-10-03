#!/usr/bin/env python3
"""
Build a production-ready video for a phone (style comes from video/builder.py).

Usage:
    python video/run_builder.py --phone-id 123 --style spec_showcase
    python video/run_builder.py --series-id 5 --style spec_showcase   # one video per phone in series
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper import db
from video.builder import build_phone_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phone-id", type=int)
    ap.add_argument("--series-id", type=int)
    ap.add_argument("--style", default="spec_showcase")
    ap.add_argument("--secs", type=int, default=10)
    ap.add_argument("--music", default=None)
    ap.add_argument("--no-intro", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="only first N phones (for testing)")
    a = ap.parse_args()

    con = db.connect()
    if a.phone_id:
        phone = db.get_phone(con, a.phone_id)
        out = build_phone_video(phone, style=a.style, secs_per_phone=a.secs,
                                intro=not a.no_intro, music=a.music)
        con.execute(
            "INSERT INTO videos (phone_id, series_id, title, file_path, status, style, created_at)"
            " VALUES (?,?,?,?, 'ready', ?, ?)",
            (a.phone_id, phone["series_id"], phone["name"], out, a.style, db.now()))
        con.commit()
        print("built:", out)
    elif a.series_id:
        from video.builder import build_series_video
        rows = con.execute(
            "SELECT p.*, b.name AS brand_name, s.name AS series_name FROM phones p "
            "JOIN brands b ON p.brand_id=b.id JOIN series s ON p.series_id=s.id "
            "WHERE p.series_id=? ORDER BY p.announced", (a.series_id,)).fetchall()
        phones = []
        for r in rows:
            d = dict(r)
            import json as _j
            d["specs"] = _j.loads(d["specs_json"] or "{}")
            phones.append(d)
        if not phones:
            print("no phones in series")
            return
        if a.limit:
            with_specs = [p for p in phones if p.get("specs")]
            phones = (with_specs or phones)[:a.limit]
        # chronological order by announced year (fallback: name)
        import re as _re
        def _yr(p):
            m = _re.search(r"(\d{4})", p.get("announced") or "")
            return (int(m.group(1)) if m else 9999, p["name"])
        phones.sort(key=_yr)
        out = build_series_video(phones, rows[0]["brand_name"], rows[0]["series_name"],
                                 style=a.style, secs_per_phone=a.secs, music=a.music)
        con.execute(
            "INSERT INTO videos (series_id, title, file_path, status, style, created_at)"
            " VALUES (?,?,?, 'ready', ?, ?)",
            (a.series_id, f"{rows[0]['brand_name']} {rows[0]['series_name']}",
             out, a.style, db.now()))
        con.commit()
        print("built:", out)
    else:
        print("give --phone-id or --series-id")


if __name__ == "__main__":
    main()
