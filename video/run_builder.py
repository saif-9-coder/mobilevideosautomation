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
    a = ap.parse_args()

    con = db.connect()
    if a.phone_id:
        ids = [a.phone_id]
    elif a.series_id:
        ids = [r["id"] for r in con.execute(
            "SELECT id FROM phones WHERE series_id=?", (a.series_id,)).fetchall()]
    else:
        print("give --phone-id or --series-id")
        return

    for pid in ids:
        phone = db.get_phone(con, pid)
        if not phone:
            print(f"phone {pid} not found")
            continue
        out = build_phone_video(phone, style=a.style)
        con.execute(
            "INSERT INTO videos (phone_id, series_id, title, file_path, status, style, created_at)"
            " VALUES (?,?,?,?, 'ready', ?, ?)",
            (pid, phone["series_id"], phone["name"], out, a.style, db.now()))
        con.commit()
        print("built:", out)


if __name__ == "__main__":
    main()
