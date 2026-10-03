#!/usr/bin/env python3
"""
MotoPhone Studio dashboard — run locally:
    pip install -r requirements.txt
    python dashboard/app.py
Then open http://127.0.0.1:5000
"""
import json
import os
import subprocess
import sys

from flask import Flask, jsonify, render_template, request, send_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scraper import db  # noqa: E402

app = Flask(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.route("/")
def index():
    con = db.connect()
    brands = con.execute(
        "SELECT b.*, COUNT(p.id) AS phones FROM brands b "
        "LEFT JOIN phones p ON p.brand_id=b.id GROUP BY b.id ORDER BY b.name"
    ).fetchall()
    return render_template("index.html", brands=brands)


@app.route("/brand/<int:brand_id>")
def brand(brand_id):
    con = db.connect()
    b = con.execute("SELECT * FROM brands WHERE id=?", (brand_id,)).fetchone()
    series = con.execute(
        "SELECT s.*, COUNT(p.id) AS phones FROM series s "
        "LEFT JOIN phones p ON p.series_id=s.id WHERE s.brand_id=? "
        "GROUP BY s.id ORDER BY s.name", (brand_id,)).fetchall()
    return render_template("brand.html", brand=b, series=series)


@app.route("/series/<int:series_id>")
def series_view(series_id):
    con = db.connect()
    s = con.execute("SELECT s.*, b.name AS brand_name FROM series s "
                    "JOIN brands b ON s.brand_id=b.id WHERE s.id=?", (series_id,)).fetchone()
    phones = con.execute("SELECT * FROM phones WHERE series_id=? ORDER BY name", (series_id,)).fetchall()
    return render_template("series.html", series=s, phones=phones)


@app.route("/phone/<int:phone_id>")
def phone_view(phone_id):
    con = db.connect()
    p = db.get_phone(con, phone_id)
    videos = con.execute("SELECT * FROM videos WHERE phone_id=? ORDER BY id DESC", (phone_id,)).fetchall()
    return render_template("phone.html", phone=p, videos=videos, specs=p["specs"])


@app.route("/img/<int:phone_id>")
def phone_img(phone_id):
    con = db.connect()
    p = con.execute("SELECT local_image FROM phones WHERE id=?", (phone_id,)).fetchone()
    if p and p["local_image"] and os.path.exists(p["local_image"]):
        return send_file(p["local_image"])
    return ("not found", 404)


@app.route("/video-file/<int:video_id>")
def video_file(video_id):
    con = db.connect()
    v = con.execute("SELECT file_path FROM videos WHERE id=?", (video_id,)).fetchone()
    if v and v["file_path"] and os.path.exists(v["file_path"]):
        return send_file(v["file_path"])
    return ("not found", 404)


@app.route("/api/build-video", methods=["POST"])
def api_build_video():
    body = request.get_json(force=True)
    phone_id = body.get("phone_id")
    style = body.get("style", "spec_showcase")
    log = os.path.join(ROOT, "data", "builder.log")
    with open(log, "a") as f:
        subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "video", "run_builder.py"),
             "--phone-id", str(phone_id), "--style", style],
            stdout=f, stderr=subprocess.STDOUT, cwd=ROOT,
        )
    con = db.connect()
    con.execute(
        "INSERT INTO videos (phone_id, title, status, style, created_at) VALUES (?,?, 'queued', ?, ?)",
        (phone_id, f"phone_{phone_id}", style, db.now()),
    )
    con.commit()
    return jsonify({"ok": True})


@app.route("/api/videos")
def api_videos():
    con = db.connect()
    rows = con.execute(
        "SELECT v.*, p.name AS phone_name FROM videos v LEFT JOIN phones p ON v.phone_id=p.id "
        "ORDER BY v.id DESC LIMIT 50").fetchall()
    return jsonify([dict(r) for r in rows])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
