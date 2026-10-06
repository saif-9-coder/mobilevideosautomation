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

# watermark-removal availability (shown as a badge in the UI so a missing
# opencv install can never silently disable cleaning)
try:
    from video.watermark import cv2_available as _wm_cv2
    WM_OK = _wm_cv2()
except Exception:
    WM_OK = False
if not WM_OK:
    print("[dashboard] WARNING: opencv not installed - watermark removal "
          "DISABLED in video builds. Run: pip install opencv-python-headless")


@app.context_processor
def _inject_wm():
    return {"wm_ok": WM_OK}
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
        p["all_images"] = [ir["local_path"] for ir in con.execute(
            "SELECT local_path FROM phone_images WHERE phone_id=? AND local_path IS NOT NULL "
            "ORDER BY position", (p["id"],)).fetchall()]
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
DATA_IMG = os.path.join(ROOT, "data", "images")


def _resolve_img(stored):
    """Resolve an image path portably: absolute path first, then basename in
    the local data/images dir (DBs copied from another machine store foreign
    absolute paths)."""
    if stored and os.path.exists(stored):
        return stored
    if stored:
        cand = os.path.join(DATA_IMG, os.path.basename(stored))
        if os.path.exists(cand):
            return cand
    return None


@app.route("/img/<int:phone_id>")
def phone_img(phone_id):
    con = db.connect()
    p = con.execute("SELECT local_image FROM phones WHERE id=?", (phone_id,)).fetchone()
    path = _resolve_img(p["local_image"]) if p else None
    if path:
        return send_file(path)
    return ("not found", 404)


@app.route("/img/<int:phone_id>/<int:pos>")
def phone_img_pos(phone_id, pos):
    con = db.connect()
    r = con.execute(
        "SELECT local_path FROM phone_images WHERE phone_id=? AND position=?",
        (phone_id, pos)).fetchone()
    path = _resolve_img(r["local_path"]) if r else None
    if path:
        return send_file(path)
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
    quality = body.get("quality", "1080p")
    con = db.connect()
    cur = con.execute(
        "INSERT INTO videos (phone_id, title, status, style, created_at) "
        "VALUES (?,?, 'queued', ?, ?)",
        (phone_id, f"phone_{phone_id}", style, db.now()))
    vid = cur.lastrowid
    con.commit()
    _launch([os.path.join(ROOT, "video", "run_builder.py"),
             "--phone-id", str(phone_id), "--video-id", str(vid),
             "--style", style, "--quality", quality])
    return jsonify({"ok": True})


@app.route("/api/build-series-video", methods=["POST"])
def api_build_series_video():
    body = request.get_json(force=True)
    series_id, style = body.get("series_id"), body.get("style", "spec_showcase")
    quality = body.get("quality", "1080p")
    con = db.connect()
    s = con.execute(
        "SELECT s.name, b.name AS bn FROM series s JOIN brands b ON s.brand_id=b.id "
        "WHERE s.id=?", (series_id,)).fetchone()
    title = f"{s['bn']} {s['name']}" if s else f"series_{series_id}"
    cur = con.execute(
        "INSERT INTO videos (series_id, title, status, style, created_at) "
        "VALUES (?,?, 'queued', ?, ?)",
        (series_id, title, style, db.now()))
    vid = cur.lastrowid
    con.commit()
    _launch([os.path.join(ROOT, "video", "run_builder.py"),
             "--series-id", str(series_id), "--video-id", str(vid),
             "--style", style, "--quality", quality])
    return jsonify({"ok": True})


@app.route("/api/build-selected-video", methods=["POST"])
def api_build_selected_video():
    import json as _j
    body = request.get_json(force=True)
    style = body.get("style", "spec_showcase")
    quality = body.get("quality", "1080p")
    if body.get("selections"):
        sel = body["selections"]
        if not sel:
            return jsonify({"ok": False, "error": "no images selected"}), 400
        n = len(sel)
        con = db.connect()
        cur = con.execute(
            "INSERT INTO videos (title, status, style, created_at) VALUES (?, 'queued', ?, ?)",
            (f"selection ({n} phones)", style, db.now()))
        vid = cur.lastrowid
        con.commit()
        _launch([os.path.join(ROOT, "video", "run_builder.py"),
                 "--selections", _j.dumps(sel), "--video-id", str(vid),
                 "--style", style, "--quality", quality])
    else:
        ids = [str(int(i)) for i in body.get("phone_ids", [])]
        if not ids:
            return jsonify({"ok": False, "error": "no phones selected"}), 400
        n = len(ids)
        con = db.connect()
        cur = con.execute(
            "INSERT INTO videos (title, status, style, created_at) VALUES (?, 'queued', ?, ?)",
            (f"selection ({n} phones)", style, db.now()))
        vid = cur.lastrowid
        con.commit()
        _launch([os.path.join(ROOT, "video", "run_builder.py"),
                 "--phone-ids", ",".join(ids), "--video-id", str(vid),
                 "--style", style, "--quality", quality])
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


EDIT_FIELDS = ["title", "released", "year", "hz", "android", "chipset",
               "display_inches", "display_panel", "camera_rear", "camera_front",
               "storage", "ram", "battery", "weight"]


@app.route("/api/phone/<int:phone_id>/edit-data")
def api_phone_edit_data(phone_id):
    from video.spec_fields import card_data
    con = db.connect()
    phone = db.get_phone(con, phone_id)
    if not phone:
        return jsonify({"ok": False}), 404
    data = card_data(phone)
    ov = phone.get("spec_overrides") or {}
    return jsonify({
        "ok": True, "id": phone_id, "name": phone["name"],
        "announced": phone.get("announced") or "",
        "fields": {k: ov.get(k) or "" for k in EDIT_FIELDS},
        "current": {
            "title": data["title"], "released": data["released"], "year": data["year"],
            "hz": data["badges"][0], "android": data["badges"][1],
            "chipset": data["badges"][2],
            "display_inches": data["rows"][0][2], "display_panel": data["rows"][0][3],
            "camera_rear": data["rows"][1][2], "camera_front": data["rows"][1][3],
            "storage": data["rows"][2][2], "ram": data["rows"][3][2],
            "battery": data["rows"][4][2], "weight": data["rows"][5][2],
        },
    })


@app.route("/api/phone/<int:phone_id>/edit", methods=["POST"])
def api_phone_edit(phone_id):
    import json as _j
    body = request.get_json(force=True)
    con = db.connect()
    if body.get("name"):
        con.execute("UPDATE phones SET name=? WHERE id=?", (body["name"], phone_id))
    if "announced" in body:
        con.execute("UPDATE phones SET announced=? WHERE id=?",
                    (body["announced"], phone_id))
    ov = {k: (body.get("fields") or {}).get(k, "") for k in EDIT_FIELDS}
    ov = {k: v for k, v in ov.items() if v}
    con.execute("UPDATE phones SET spec_overrides=? WHERE id=?",
                (_j.dumps(ov, ensure_ascii=False), phone_id))
    con.commit()
    # drop cached preview frames so the edit shows immediately
    import glob as _g
    for f in _g.glob(os.path.join(FRAMES, f"preview_{phone_id}_*")):
        try:
            os.remove(f)
        except OSError:
            pass
    return jsonify({"ok": True})


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


@app.route("/eraser")
def eraser_page():
    """Magic Eraser — Canva-style manual brush eraser."""
    return render_template("eraser.html")


@app.route("/api/eraser", methods=["POST"])
def api_eraser():
    """Inpaint the user-brushed mask. Only masked pixels are blended;
    everything else stays bit-identical (v12 approach)."""
    import base64
    import io
    try:
        import cv2
        import numpy as np
    except ImportError:
        return jsonify({"ok": False, "error": "opencv not installed"}), 500
    try:
        data = request.get_json(force=True)
        img_b64 = data["image"].split(",", 1)[-1]
        mask_b64 = data["mask"].split(",", 1)[-1]
        img_arr = np.frombuffer(base64.b64decode(img_b64), np.uint8)
        mask_arr = np.frombuffer(base64.b64decode(mask_b64), np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        mask_raw = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)
        if img is None or mask_raw is None:
            return jsonify({"ok": False, "error": "bad image/mask"}), 400
        h, w = img.shape[:2]
        mask = cv2.resize(mask_raw, (w, h))
        _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
        if int((mask > 0).sum()) < 10:
            return jsonify({"ok": False, "error": "brush over the area first"}), 400
        # Dilate once so stroke edges are fully covered
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        # Telea inpaint, then blend ONLY masked pixels (feathered edges)
        filled = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        m = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (5, 5), 0)
        m3 = np.stack([m] * 3, axis=2)
        out = (img.astype(np.float32) * (1 - m3) + filled.astype(np.float32) * m3)
        out = np.clip(out, 0, 255).astype(np.uint8)
        _, buf = cv2.imencode(".png", out)
        out_b64 = base64.b64encode(buf).decode()
        return jsonify({"ok": True, "image": "data:image/png;base64," + out_b64})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
