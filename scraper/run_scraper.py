#!/usr/bin/env python3
"""
CLI for the GSMArena scraping pipeline.

Usage:
    python run_scraper.py brands                 # scrape all brands into DB
    python run_scraper.py phones --brand Samsung  # scrape phone list for a brand
    python run_scraper.py phones --all            # scrape phone lists for ALL brands
    python run_scraper.py specs --brand Samsung   # scrape full specs for a brand's phones
    python run_scraper.py specs --phone-id 123    # scrape full specs for one phone
    python run_scraper.py images --brand Samsung  # download phone images locally

Run repeatedly (e.g. via cron) to keep data fresh:
    0 3 * * * cd /path/to/mobilevideosautomation && python scraper/run_scraper.py specs --all >> data/scraper.log 2>&1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper import db, gsmarena

DATA_IMG = os.path.join(os.path.dirname(__file__), "..", "data", "images")


def cmd_brands(con):
    brands = gsmarena.scrape_brands()
    for b in brands:
        db.upsert_brand(con, b["name"], b["url"])
    print(f"saved {len(brands)} brands")


def _brand_ids(con, brand):
    rows = con.execute(
        "SELECT id, name, gsmarena_url FROM brands WHERE name LIKE ?", (f"%{brand}%",)
    ).fetchall()
    return rows


def cmd_phones(con, brand=None, all_brands=False):
    brands = (
        con.execute("SELECT id, name, gsmarena_url FROM brands").fetchall()
        if all_brands
        else _brand_ids(con, brand)
    )
    total = 0
    for br in brands:
        if not br["gsmarena_url"]:
            continue
        phones = gsmarena.scrape_brand_phones(br["gsmarena_url"])
        for p in phones:
            series = db.guess_series(p["name"])
            sid = db.get_or_create_series(con, br["id"], series)
            db.upsert_phone(con, br["id"], sid, p["name"], p["url"], image_url=p.get("img"))
        total += len(phones)
        print(f"{br['name']}: {len(phones)} phones")
    print(f"total {total} phones")


def cmd_specs(con, brand=None, all_brands=False, phone_id=None, series_id=None,
              missing_only=False):
    if phone_id:
        rows = con.execute("SELECT id, gsmarena_url FROM phones WHERE id=?", (phone_id,)).fetchall()
    elif series_id:
        rows = con.execute(
            "SELECT id, gsmarena_url, specs_json FROM phones WHERE series_id=?",
            (series_id,)).fetchall()
        if missing_only:
            import json as _j
            rows = [r for r in rows if not _j.loads(r["specs_json"] or "{}")]
    elif all_brands:
        rows = con.execute("SELECT id, gsmarena_url FROM phones").fetchall()
    else:
        rows = con.execute(
            "SELECT p.id, p.gsmarena_url FROM phones p JOIN brands b ON p.brand_id=b.id "
            "WHERE b.name LIKE ?", (f"%{brand}%",)
        ).fetchall()
    for i, r in enumerate(rows):
        data = gsmarena.scrape_phone_specs(r["gsmarena_url"])
        if data:
            specs = data.get("specs", {})
            announced = specs.get("Launch", {}).get("Announced", "")
            status = specs.get("Launch", {}).get("Status", "")
            con.execute(
                "UPDATE phones SET specs_json=?, announced=?, status=?, scraped_at=? WHERE id=?",
                (__import__("json").dumps(specs, ensure_ascii=False), announced, status,
                 db.now(), r["id"]),
            )
            con.commit()
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(rows)}")
    print(f"specs updated for {len(rows)} phones")


def cmd_images(con, brand=None, series_id=None, missing_only=True):
    import requests

    os.makedirs(DATA_IMG, exist_ok=True)
    if series_id:
        rows = con.execute(
            "SELECT id, name, image_url, local_image FROM phones WHERE series_id=?",
            (series_id,)).fetchall()
    else:
        q = "SELECT id, name, image_url, local_image FROM phones WHERE local_image IS NULL"
        params: tuple = ()
        if brand:
            q = ("SELECT p.id, p.name, p.image_url, p.local_image FROM phones p "
                 "JOIN brands b ON p.brand_id=b.id "
                 "WHERE b.name LIKE ? AND p.local_image IS NULL")
            params = (f"%{brand}%",)
        rows = con.execute(q, params).fetchall()
    if missing_only:
        rows = [r for r in rows if not r["local_image"] or not os.path.exists(r["local_image"])]
    for r in rows:
        if not r["image_url"]:
            continue
        ext = os.path.splitext(r["image_url"].split("?")[0])[1] or ".jpg"
        path = os.path.join(DATA_IMG, f"phone_{r['id']}{ext}")
        try:
            img = requests.get(r["image_url"], headers=gsmarena.UA, timeout=30).content
            if len(img) > 2000:
                open(path, "wb").write(img)
                con.execute("UPDATE phones SET local_image=? WHERE id=?", (path, r["id"]))
                con.commit()
        except Exception:
            pass
    print(f"downloaded images, {len(rows)} candidates")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("brands")
    p = sub.add_parser("phones")
    p.add_argument("--brand"); p.add_argument("--all", action="store_true")
    p = sub.add_parser("specs")
    p.add_argument("--brand"); p.add_argument("--all", action="store_true")
    p.add_argument("--phone-id", type=int); p.add_argument("--series-id", type=int)
    p.add_argument("--missing-only", action="store_true")
    p = sub.add_parser("images")
    p.add_argument("--brand"); p.add_argument("--series-id", type=int)
    a = ap.parse_args()

    con = db.connect()
    if a.cmd == "brands":
        cmd_brands(con)
    elif a.cmd == "phones":
        cmd_phones(con, a.brand, a.all)
    elif a.cmd == "specs":
        cmd_specs(con, a.brand, a.all, a.phone_id, a.series_id, a.missing_only)
    elif a.cmd == "images":
        cmd_images(con, a.brand, a.series_id)


if __name__ == "__main__":
    main()
