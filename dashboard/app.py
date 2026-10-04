#!/usr/bin/env python3
"""
MotoPhone Studio dashboard — run locally:
    pip install -r requirements.txt
    python dashboard/app.py
Then open http://127.0.0.1:5000

Pages:
    /               Home — brands, each with its series listed series-wise
    /brand/<id>     Brand — series cards
    /series/<id>    Series — phone cards (image + specs + preview slideshow + build buttons)
    /manage         Manage — all videos: watch, select & delete, build status
"""
import json
import os
import subprocess
import sys

from flask import Flask, jsonify, render_template, request, send_file, Response

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scraper import db  # noqa: E402

app = Flask(__name__)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES = os.path.join(ROOT, "data", "frames")


def _phone_dict(row):
    d = dict(row)
    d["specs"] = json.loads(d.get("specs_json") or "{}")
    return d


def _key_specs(phone):
    s = phone["specs"]
    g = lambda sec, k: (s.get(sec, {}) or {}).get(k, "")
    disp = g("Display", "Size").split(",")[0]
    chip = g("Platform", "Chipset")
    chip = chip.split("(")[0].strip()
    if len(chip) > 34:
        chip = chip[:34] + "…"
    return {
        "display": disp, "chipset": chip,
        "ram": g("Memory", "Internal").split(",")[0][:24],
        "battery": g("Battery", "Type")[:30],
        "camera": (g("Main Camera", "Triple") or g("Main Camera", "Quad") or "")[:30],
    }


# ---------------- pages ----------------
@app.route("/")
def index():
    con = db.connect()
    brands = con.execute(
        "SELECT b.*, COUNT(p.id) AS phones FROM brands b "
        "LEFT JOIN phones p ON p.brand_id=b.id GROUP BY b.id ORDER BY b.name").fetchall()
    out = []
    for b in brands:
        series = con.execute(
            "SELECT s.id, s.name, COUNT(p.id) AS phones FROM series s "
            "LEFT JOIN phones p ON p.series_id=s.id WHERE s.brand_id=? "
            "GROUP BY s.id HAVING phones > 0 ORDER BY phones DESC, s.name LIMIT 12",
            (b["id"],)).fetchall()
        out.append({"brand": b, "series": series})
    videos = con.execute(
        "SELECT v.*, p.name AS phone_name FROM videos v "
        "LEFT JOIN phones p ON v.phone_id=p.id ORDER BY v.id DESC LIMIT 8").fetchall()
    return render_template("index.html", brands=out, videos=videos)


@app.route("/brand/<int:brand_id>")
def brand_view(brand_id):
    con = db.connect()
    b = con.execute("SELECT * FROM brands WHERE id=?", (brand_id,)).fetchone()
    series = con.execute(
        "SELECT s.*, COUNT(p.id) AS phones FROM series s "
        "LEFT JOIN phones p ON p.series_id=s.id WHERE s.brand_id=? "
        "GROUP BY s.id HAVING phones > 0 ORDER BY phones DESC, s.name",
        (brand_id,)).fetchall()
    return render_template("brand.html", brand=b, series=series)


@app.route("/series/<int:series_id>")
def series_view(series_id):
    con = db.connect()
    s = con.execute(
        "SELECT s.*, b.name AS brand_name, b.id AS brand_id FROM series s "
        "JOIN brands b ON s.brand_id=b.id WHERE s.id=?", (series_id,)).fetchone()
    rows = con.execute(
        "SELECT * FROM phones WHERE series_id=? ORDER BY announced, name",
        (series_id,)).fetchall()
    phones = []
    for r in rows:
        p = _phone_dict(r)
        p["key"] = _key_specs(p)
        phones.append(p)
    videos = con.execute(
        "SELECT * FROM videos WHERE series_id=? ORDER BY id DESC", (series_id,)).fetchall()
    return render_template("series.html", series=s, phones=phones, videos=videos)


@app.route("/manage")
def manage():
    con = db.connect()
    videos = con.execute(
        "SELECT v.*, p.name AS phone_name, s.name AS series_name FROM videos v "
        "LEFT JOIN phones p ON v.phone_id=p.id "
        "LEFT JOIN series s ON v.series_id=s.id ORDER BY v.id DESC").fetchall()
    return render_template("manage.html", videos=videos)


# ---------------- phone assets ----------------
@app.route("/img/<int:phone_id>")
def phone_img(phone_id):
    con = db.connect()
    p = con.execute("SELECT local_image FROM phones WHERE id=?", (phone_id,)).fetchone()
    if p and p["local_image"] and os.path.exists(p["local_image"]):
        return send_file(p["local_image"])
    return ("not found", 404)


@app.route("/slide/<int:phone_id>/<int:idx>")
def slide(phone_id, idx):
    """Render preview slides on demand: 0 = intro card, 1 = phone spec card."""
    from video.spec_fields import card_data
    from video.style_spec import render_card, render_intro
    from video.builder import prepare_image

    theme = request.args.get("theme", "spec_showcase")
    con = db.connect()
    phone = db.get_phone(con, phone_id)
    if not phone:
        return ("not found", 404)
    os.makedirs(FRAMES, exist_ok=True)
    out = os.path.join(FRAMES, f"preview_{phone_id}_{idx}_{theme}.png")
    if not os.path.exists(out):
        data = card_data(phone)
        img_path = prepare_image(phone)
        if idx == 0:
            render_intro(phone["name"].split()[0], data["title"], out,
                         [img_path] if img_path else None, theme=theme)
        else:
            render_card(data, img_path, out, theme=theme)
    return send_file(out)


@app.route("/video-file/<int:video_id>")
def video_file(video_id):
    con = db.connect()
    v = con.execute("SELECT file_path FROM videos WHERE id=?", (video_id,)).fetchone()
    if v and v["file_path"] and os.path.exists(v["file_path"]):
        return send_file(v["file_path"])
    return ("not found", 404)


# ---------------- actions ----------------
def _launch(args):
    log = os.path.join(ROOT, "data", "builder.log")
    with open(log, "a") as f:
        subprocess.Popen([sys.executable] + args, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)


@app.route("/api/build-video", methods=["POST"])
def api_build_video():
    body = request.get_json(force=True)
    phone_id, style = body.get("phone_id"), body.get("style", "spec_showcase")
    _launch([os.path.join(ROOT, "video", "run_builder.py"),
             "--phone-id", str(phone_id), "--style", style])
    con = db.connect()
    con.execute(
        "INSERT INTO videos (phone_id, title, status, style, created_at) "
        "VALUES (?,?, 'queued', ?, ?)",
        (phone_id, f"phone_{phone_id}", style, db.now()))
    con.commit()
    return jsonify({"ok": True})


@app.route("/api/build-series-video", methods=["POST"])
def api_build_series_video():
    body = request.get_json(force=True)
    series_id, style = body.get("series_id"), body.get("style", "spec_showcase")
    _launch([os.path.join(ROOT, "video", "run_builder.py"),
             "--series-id", str(series_id), "--style", style])
    con = db.connect()
    s = con.execute(
        "SELECT s.name, b.name AS bn FROM series s JOIN brands b ON s.brand_id=b.id "
        "WHERE s.id=?", (series_id,)).fetchone()
    con.execute(
        "INSERT INTO videos (series_id, title, status, style, created_at) "
        "VALUES (?,?, 'queued', ?, ?)",
        (series_id, f"{s['bn']} {s['name']}" if s else f"series_{series_id}",
         style, db.now()))
    con.commit()
    return jsonify({"ok": True})


@app.route("/api/build-selected-video", methods=["POST"])
def api_build_selected_video():
    body = request.get_json(force=True)
    ids = [str(int(i)) for i in body.get("phone_ids", [])]
    style = body.get("style", "spec_showcase")
    if not ids:
        return jsonify({"ok": False, "error": "no phones selected"}), 400
    _launch([os.path.join(ROOT, "video", "run_builder.py"),
             "--phone-ids", ",".join(ids), "--style", style])
    con = db.connect()
    con.execute(
        "INSERT INTO videos (title, status, style, created_at) VALUES (?, 'queued', ?, ?)",
        (f"selection ({len(ids)} phones)", style, db.now()))
    con.commit()
    return jsonify({"ok": True})


@app.route("/api/scrape-specs", methods=["POST"])
def api_scrape_specs():
    body = request.get_json(force=True)
    series_id = int(body["series_id"])
    _launch([os.path.join(ROOT, "scraper", "run_scraper.py"),
             "specs", "--series-id", str(series_id), "--missing-only"])
    return jsonify({"ok": True})


@app.route("/api/scrape-images", methods=["POST"])
def api_scrape_images():
    body = request.get_json(force=True)
    series_id = int(body["series_id"])
    _launch([os.path.join(ROOT, "scraper", "run_scraper.py"),
             "images", "--series-id", str(series_id)])
    return jsonify({"ok": True})


@app.route("/api/templates")
def api_templates():
    from video.style_spec import TEMPLATE_CHOICES
    return jsonify([{"id": t[0], "name": t[1]} for t in TEMPLATE_CHOICES])


@app.route("/api/delete-videos", methods=["POST"])
def api_delete_videos():
    body = request.get_json(force=True)
    ids = [int(i) for i in body.get("ids", [])]
    con = db.connect()
    n = 0
    for vid in ids:
        v = con.execute("SELECT file_path FROM videos WHERE id=?", (vid,)).fetchone()
        if v:
            if v["file_path"] and os.path.exists(v["file_path"]):
                try:
                    os.remove(v["file_path"])
                except OSError:
                    pass
            con.execute("DELETE FROM videos WHERE id=?", (vid,))
            n += 1
    con.commit()
    return jsonify({"ok": True, "deleted": n})


@app.route("/api/videos")
def api_videos():
    con = db.connect()
    rows = con.execute(
        "SELECT v.*, p.name AS phone_name FROM videos v "
        "LEFT JOIN phones p ON v.phone_id=p.id ORDER BY v.id DESC LIMIT 50").fetchall()
    return jsonify([dict(r) for r in rows])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
