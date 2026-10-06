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
    """Smart Magic Eraser (Canva-style).

    The user brushes over the object/watermark to remove. Instead of
    inpainting the whole brushed blob (which smears the area), we detect
    the actual foreground strokes inside the brush — pixels that differ
    from the local background — and inpaint only those with a tight mask.
    Everything else stays bit-identical. For non-text objects (no strokes
    found), the whole brush is inpainted with a gentle edge.
    """
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

        brush = (mask > 0).astype(np.uint8)
        # --- Smart step: find foreground strokes inside the brush ---
        # Local background estimate (large median blur removes thin text)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bg = cv2.medianBlur(gray, 21)
        dev = cv2.absdiff(gray, bg)
        brush_search = cv2.dilate(brush,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                                  iterations=1)
        strokes = ((dev > 14).astype(np.uint8)) * brush
        strokes = cv2.morphologyEx(strokes, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        stroke_frac = float(strokes.sum()) / max(1, int(brush.sum()))

        if int(strokes.sum()) > 40 and stroke_frac < 0.7:
            # Text/watermark: tight stroke mask
            sm = cv2.dilate(strokes * 255,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                            iterations=1)
            # If background around strokes is smooth, fill with median color
            # (cleanest on gradients — no inpaint blur). Else Navier-Stokes.
            ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            ring = cv2.subtract(cv2.dilate(sm, ring_k, iterations=2), sm)
            ring_px = img[ring > 0]
            use_flat = False
            if len(ring_px) > 50:
                if float(ring_px.reshape(-1, 3).std()) < 18:
                    use_flat = True
            if use_flat:
                med = np.median(ring_px.reshape(-1, 3), axis=0).astype(np.uint8)
                filled = img.copy()
                filled[sm > 0] = med
            else:
                # Try LaMa AI first (Canva-like quality) if available
                filled = None
                try:
                    from simple_lama_inpainting import SimpleLama
                    from PIL import Image
                    lama = SimpleLama()
                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    pil_m = Image.fromarray(sm)
                    mw, mh = pil_img.size
                    sc = min(1.0, 1024 / max(mw, mh))
                    if sc < 1.0:
                        rs = lama(pil_img.resize((int(mw*sc), int(mh*sc)), Image.LANCZOS),
                                  pil_m.resize((int(mw*sc), int(mh*sc)), Image.LANCZOS))
                        res = rs.resize((mw, mh), Image.LANCZOS)
                    else:
                        res = lama(pil_img, pil_m)
                    filled = cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR)
                except Exception:
                    filled = None
                if filled is None:
                    filled = cv2.inpaint(img, sm, 4, cv2.INPAINT_NS)
            mf = cv2.GaussianBlur(sm.astype(np.float32) / 255.0, (3, 3), 0)
        else:
            # Object: whole brush, gentle edge, Navier-Stokes inpaint
            m = cv2.dilate(brush * 255,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                           iterations=1)
            filled = cv2.inpaint(img, m, 4, cv2.INPAINT_NS)
            mf = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (5, 5), 0)
        m3 = np.stack([mf] * 3, axis=2)
        out = (img.astype(np.float32) * (1 - m3) + filled.astype(np.float32) * m3)
        out = np.clip(out, 0, 255).astype(np.uint8)
        _, buf = cv2.imencode(".png", out)
        out_b64 = base64.b64encode(buf).decode()
        return jsonify({"ok": True, "image": "data:image/png;base64," + out_b64})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
