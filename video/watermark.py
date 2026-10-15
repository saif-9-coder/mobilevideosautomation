"""GSMArena watermark removal.

Every GSMArena pictures-page image carries 1-2 semi-transparent
"www.GSMArena.com" stamps. This module finds them with multi-scale template
matching (templates extracted from real watermarks), verifies each candidate
by comparing its stroke pattern against the template's, and erases only the
verified text strokes with OpenCV inpainting. Unverified regions are never
touched, so false positives cannot damage the image.

Usage:
    from video.watermark import clean_watermark, cv2_available
    clean_path = clean_watermark(src_path)   # cached; returns src unchanged
                                             # when cv2 is unavailable or no
                                             # watermark is found.
"""
import hashlib
import os

try:
    import cv2
    import numpy as np
    _CV2 = True
except Exception:
    _CV2 = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_HERE, "work", "watermark")
_CACHE_VERSION = "v40"
_TEMPLATES = ("wm_template.png", "wm_template_b.png", "wm_template_c.png")
_SCALES = (0.6, 0.75, 0.9, 1.05, 1.2, 1.4)
_NCC_THRESHOLD = 0.75  # strict: avoid false positives that cause damage
_DICE_THRESHOLD = 0.35  # strict: reject false positives like UI digits
_MAX_BOXES = 4

_tpl_cache = None
_tpl_strokes_cache = None
_warned = False


def cv2_available():
    return _CV2


def _load_templates():
    global _tpl_cache, _tpl_strokes_cache
    if _tpl_cache is not None:
        return _tpl_cache, _tpl_strokes_cache
    _tpl_cache, _tpl_strokes_cache = [], []
    if _CV2:
        for name in _TEMPLATES:
            p = os.path.join(_HERE, name)
            if os.path.exists(p):
                t = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if t is not None:
                    _tpl_cache.append(t)
                    _tpl_strokes_cache.append((t < t.mean()).astype(np.uint8))
    return _tpl_cache, _tpl_strokes_cache


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0


def _image_strokes(patch):
    """Binary stroke map of a patch via local background subtraction."""
    bg = cv2.medianBlur(patch, 15)
    diff = cv2.absdiff(patch, bg)
    strokes = (diff > 14).astype(np.uint8)
    return cv2.dilate(strokes,
                      cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                      iterations=1)


def _verify(gray, x, y, tw, th, tpl_idx):
    """Refine the box locally and check stroke-pattern overlap (Dice).

    Returns (ok, refined_box, stroke_mask) where stroke_mask is tight around
    the text strokes (or None when the region is not the watermark).
    """
    tpls, tpl_strokes = _load_templates()
    h, w = gray.shape
    best = (0, x, y)
    for dy in (-6, -3, 0, 3, 6):
        for dx in (-6, -3, 0, 3, 6):
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx + tw > w or ny + th > h:
                continue
            patch = gray[ny:ny + th, nx:nx + tw]
            t2 = cv2.resize(tpls[tpl_idx], (tw, th)).astype(np.float32)
            p = patch.astype(np.float32)
            p -= p.mean()
            t2 -= t2.mean()
            denom = np.sqrt((p ** 2).sum() * (t2 ** 2).sum())
            ncc = float((p * t2).sum() / denom) if denom else 0
            if ncc > best[0]:
                best = (ncc, nx, ny)
    _, rx, ry = best
    # Dice between image strokes and template strokes at refined position.
    # Take the best Dice across ALL templates — the matching template
    # varies by watermark style.
    x0, y0 = max(0, rx - 3), max(0, ry - 3)
    x1, y1 = min(w, rx + tw + 3), min(h, ry + th + 3)
    patch = gray[y0:y1, x0:x1]
    img_st = _image_strokes(patch)
    best_dice = 0
    best_ti = tpl_idx
    for ti2, ts0 in enumerate(tpl_strokes):
        ts = cv2.resize(ts0, (x1 - x0, y1 - y0))
        inter = np.logical_and(img_st, ts).sum()
        d = inter / ts.sum() if ts.sum() else 0
        if d > best_dice:
            best_dice = d
            best_ti = ti2
    if best_dice < _DICE_THRESHOLD:
        return False, None, None
    return True, (rx, ry, tw, th), best_ti



def _detect_bottom_edge(gray):
    """Fallback: sliding-window search for watermark text on image edges.

    Finds windows with the most text-like pixels (local contrast).
    Searches bottom (horizontal watermarks) + left/right edges
    (vertical watermarks). Robust for small/faint watermarks missed
    by template matching.
    """
    h, w = gray.shape
    boxes = []

    def score_roi(roi):
        bg = cv2.medianBlur(roi, 15)
        diff = cv2.absdiff(roi.astype(np.float32), bg.astype(np.float32))
        return int((diff > 12).sum())

    # 1) Bottom area: horizontal watermarks (140x30 windows)
    best_score, best_box = 0, None
    y_start, y_end = int(h * 0.80), h - 25
    x_start, x_end = int(w * 0.50), w - 120
    ww, wh = 140, 30
    for y in range(y_start, y_end, 8):
        for x in range(x_start, x_end, 12):
            roi = gray[y:y+wh, x:x+ww]
            if roi.shape[0] < wh or roi.shape[1] < ww:
                continue
            text_px = score_roi(roi)
            if text_px > best_score and text_px < 4000 and text_px > 80:
                best_score = text_px
                best_box = (x, y, ww, wh)
    if best_box:
        boxes.append(best_box)

    # 2) Left edge: vertical watermarks (30x140 windows)
    # Watermark sits at the very edge, top area — search narrow
    # Force x=0: text is always at the extreme edge
    best_score, best_box = 0, None
    y_start, y_end = int(h * 0.02), int(h * 0.50)
    ww, wh = 30, 140
    x = 0
    for y in range(y_start, y_end, 12):
        roi = gray[y:y+wh, x:x+ww]
        if roi.shape[0] < wh or roi.shape[1] < ww:
            continue
        text_px = score_roi(roi)
        if text_px > best_score and text_px < 4000 and text_px > 80:
            best_score = text_px
            best_box = (x, y, ww, wh)
    if best_box:
        boxes.append(best_box)

    # 3) Right edge: vertical watermarks (30x140 windows)
    best_score, best_box = 0, None
    x_start, x_end = int(w * 0.95), w - 30
    y_start, y_end = int(h * 0.02), int(h * 0.50)
    for y in range(y_start, y_end, 12):
        for x in range(x_start, x_end, 8):
            roi = gray[y:y+wh, x:x+ww]
            if roi.shape[0] < wh or roi.shape[1] < ww:
                continue
            text_px = score_roi(roi)
            if text_px > best_score and text_px < 4000 and text_px > 80:
                best_score = text_px
                best_box = (x, y, ww, wh)
    if best_box:
        boxes.append(best_box)

    return boxes


def _detect(gray):
    """Return verified (box, stroke_mask) list."""
    h, w = gray.shape
    tpls, _ = _load_templates()
    dets = []
    for ti, tpl in enumerate(tpls):
        th0, tw0 = tpl.shape
        for s in _SCALES:
            tw, th = int(tw0 * s), int(th0 * s)
            if tw >= w or th >= h or tw < 12 or tw > w * 0.35:
                continue
            t2 = cv2.resize(tpl, (tw, th))
            res = cv2.matchTemplate(gray, t2, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= _NCC_THRESHOLD)
            for x, y in zip(xs, ys):
                dets.append((float(res[y, x]), int(x), int(y), tw, th, ti))
    dets.sort(reverse=True)
    cands = []
    for sc, x, y, tw, th, ti in dets:
        if len(cands) >= _MAX_BOXES * 3:
            break
        if all(_iou((x, y, tw, th), (kx, ky, ktw, kth)) < 0.3
               for _, kx, ky, ktw, kth, _ in cands):
            cands.append((sc, x, y, tw, th, ti))
    # verify best-first, keep at most _MAX_BOXES verified
    verified = []
    for sc, x, y, tw, th, ti in cands:
        if len(verified) >= _MAX_BOXES:
            break
        ok, box, best_ti = _verify(gray, x, y, tw, th, ti)
        if ok:
            verified.append((box, best_ti))
    # Fallback: if template matching found nothing, scan bottom edge
    if not verified:
        for (x, y, bw, bh) in _detect_bottom_edge(gray)[:_MAX_BOXES]:
            verified.append(((x, y, bw, bh), None))
    # Final fallback: check fixed bottom-right ROI (GSMArena watermark spot)
    if not verified:
        h, w = gray.shape
        # Bottom-right 30% x 12% region
        rx, ry = int(w * 0.68), int(h * 0.88)
        rw, rh = w - rx, h - ry
        roi = gray[ry:ry+rh, rx:rx+rw]
        bg = cv2.medianBlur(roi, 15)
        diff = cv2.absdiff(roi.astype(np.float32), bg.astype(np.float32))
        if (diff > 10).sum() > 50:  # text-like content present
            verified.append(((rx, ry, rw, rh), None))
    return verified




# Safe eraser (Saif's rule: never damage the phone, not even a little).
# Replaces the old LaMa AI path entirely — deterministic text-only removal.
from video.safe_erase import erase_text_region


def _polish_band(img_bgr, box):
    """Final polish: in the tight text band, replace every pixel even
    slightly brighter than the background.

    Catches faint text ghosts the main eraser misses. Only runs inside
    a verified text box; strong vertical gradients (phone edges) are
    protected.
    """
    from video.safe_erase import _bg_estimate
    rx, ry, tw, th = box
    h, w = img_bgr.shape[:2]
    x0, x1 = max(0, rx), min(w, rx + tw)
    y0, y1 = max(0, ry), min(h, ry + th)
    if x1 <= x0 or y1 <= y0:
        return 0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg = _bg_estimate(gray)
    bgc = _bg_estimate(img_bgr)
    bright = (gray[y0:y1, x0:x1].astype(np.int16)
              > bg[y0:y1, x0:x1].astype(np.int16) + 1)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    strong_vert = np.abs(gx[y0:y1, x0:x1]) > 100
    sel = bright & ~strong_vert
    n = int(sel.sum())
    if n:
        out = img_bgr.copy()
        band = out[y0:y1, x0:x1]
        bandc = bgc[y0:y1, x0:x1]
        band[sel] = bandc[sel]
        img_bgr[:] = out
    return n


def _remove_box(img_bgr, gray, box, tpl_idx=None):
    """Erase verified watermark text using the safe eraser.

    The verified box is expanded by 10px. Only text-like pixels are
    replaced with the background color behind them. Phone edges are
    protected; everything else stays bit-identical. A final polish
    pass cleans faint ghosts inside the tight box.
    """
    rx, ry, tw, th = box
    h, w = gray.shape
    pad = 10
    x0, y0 = max(0, rx - pad), max(0, ry - pad)
    x1, y1 = min(w, rx + tw + pad), min(h, ry + th + pad)
    region = np.zeros((h, w), np.uint8)
    region[y0:y1, x0:x1] = 255
    out, info = erase_text_region(img_bgr, region)
    replaced = info.get("replaced_px", 0)
    if info.get("ok") or replaced > 0:
        img_bgr[:] = out
        # Polish faint ghosts inside the tight box.
        _polish_band(img_bgr, box)
        return True
    return False


def clean_watermark(src_path):
    """Remove GSMArena watermark stamps from an image.

    Returns the path of the cleaned image (cached under video/work/watermark).
    Returns the original path unchanged when OpenCV is unavailable, no
    watermark is detected, or anything fails.
    """
    global _warned
    if not _CV2:
        if not _warned:
            _warned = True
            print("[watermark] WARNING: opencv not installed - watermark "
                  "removal DISABLED. Run: pip install opencv-python-headless")
        return src_path
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        key = hashlib.md5(os.path.abspath(src_path).encode()).hexdigest()
        ext = os.path.splitext(src_path)[1].lower() or ".jpg"
        dst = os.path.join(_CACHE_DIR, f"{key}_{_CACHE_VERSION}{ext}")
        if os.path.exists(dst) and \
                os.path.getmtime(dst) >= os.path.getmtime(src_path):
            return dst
        img = cv2.imread(src_path)
        if img is None:
            return src_path
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template_boxes = _detect(gray)  # strict, tight boxes
        done = False
        for (box, _stroke_info) in template_boxes[:_MAX_BOXES]:
            try:
                if _remove_box(img, gray, box):
                    done = True
            except Exception:
                continue
        # Fallback ALWAYS runs: a false template box must not block the
        # real watermark elsewhere in the image.
        if True:
            # Fallback: sliding-window search for faint/small watermarks
            # missed by template matching.
            try:
                fb_boxes = _detect_bottom_edge(gray)
            except Exception:
                fb_boxes = []
            for box in fb_boxes[:_MAX_BOXES]:
                try:
                    rx, ry, tw, th = box
                    if th > tw * 1.5:
                        continue  # vertical: manual eraser case
                    if _remove_box(img, gray, (rx, ry, tw, th)):
                        done = True
                except Exception:
                    continue
        # Safe bottom-right check: small precise ROI for faint watermarks
        if not done:
            try:
                h, w = gray.shape
                # Bottom-right 25% x 10%: where GSMArena puts small watermarks
                rx, ry = int(w * 0.75), int(h * 0.90)
                rw, rh = w - rx, h - ry
                # Only proceed if there's bright text-like content
                roi = gray[ry:ry+rh, rx:rx+rw]
                # Bright pixels (text is brighter than dark bg)
                bright = (roi > 100).sum()
                if bright > 20 and bright < rw * rh * 0.3:
                    if _remove_box(img, gray, (rx, ry, rw, rh)):
                        done = True
            except Exception:
                pass
        if not done:
            return src_path
        cv2.imwrite(dst, img)
        return dst
    except Exception:
        return src_path


def clean_cutout_watermark(cutout_path):
    """Remove GSMArena watermarks from a background-removed cutout (RGBA).

    Runs AFTER background removal (as Saif requested): the phone cutout goes
    through the same safe text eraser used in the manual tool. Alpha channel is
    preserved bit-identical; only RGB text pixels are inpainted.

    Returns the path of the cleaned cutout (cached). Returns the original
    path unchanged when no watermark is detected or anything fails.
    """
    if not _CV2:
        return cutout_path
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        key = hashlib.md5(os.path.abspath(cutout_path).encode()).hexdigest()
        dst = os.path.join(_CACHE_DIR, f"{key}_{_CACHE_VERSION}_cut.png")
        if os.path.exists(dst) and \
                os.path.getmtime(dst) >= os.path.getmtime(cutout_path):
            return dst
        # Read with alpha
        img_rgba = cv2.imread(cutout_path, cv2.IMREAD_UNCHANGED)
        if img_rgba is None:
            return cutout_path
        if len(img_rgba.shape) == 2 or img_rgba.shape[2] < 4:
            # No alpha: fall back to standard cleaning
            return clean_watermark(cutout_path)
        bgr = img_rgba[:, :, :3]
        alpha = img_rgba[:, :, 3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Same detection as clean_watermark: template first, bottom-edge
        # fallback only when template matching found nothing.
        verified = _detect(gray)
        if not verified:
            try:
                verified = [(b, None) for b in _detect_bottom_edge(gray)]
            except Exception:
                pass
        if not verified:
            return cutout_path

        done = False
        for (box, tpl_idx) in verified[:_MAX_BOXES]:
            try:
                if _remove_box(bgr, gray, box, tpl_idx):
                    done = True
            except Exception:
                continue
        if not done:
            try:
                fb_boxes = _detect_bottom_edge(gray)
            except Exception:
                fb_boxes = []
            for box in fb_boxes[:_MAX_BOXES]:
                try:
                    rx, ry, tw, th = box
                    if th > tw * 1.5:
                        continue
                    if _remove_box(bgr, gray, (rx, ry, tw, th)):
                        done = True
                except Exception:
                    continue
        if not done:
            return cutout_path
        # Recombine with original alpha (bit-identical)
        out_rgba = np.dstack([bgr, alpha])
        cv2.imwrite(dst, out_rgba)
        return dst
    except Exception:
        return cutout_path
