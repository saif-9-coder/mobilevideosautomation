"""GSMArena watermark removal.

Every GSMArena pictures-page image carries 1-2 semi-transparent
"www.GSMArena.com" stamps. This module finds them with multi-scale template
matching (templates extracted from real watermarks) and erases them with
OpenCV Navier-Stokes inpainting, so the filled area blends seamlessly with
the surrounding pixels.

Usage:
    from video.watermark import clean_watermark
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
_TEMPLATES = ("wm_template.png", "wm_template_b.png")
_SCALES = (0.65, 0.8, 0.95, 1.1, 1.25, 1.4)
_THRESHOLD = 0.80

_tpl_cache = None


def _load_templates():
    global _tpl_cache
    if _tpl_cache is not None:
        return _tpl_cache
    _tpl_cache = []
    if not _CV2:
        return _tpl_cache
    for name in _TEMPLATES:
        p = os.path.join(_HERE, name)
        if os.path.exists(p):
            t = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if t is not None:
                _tpl_cache.append(t)
    return _tpl_cache


def _detect(gray):
    """Return list of (x, y, w, h) watermark boxes."""
    h, w = gray.shape
    dets = []
    for tpl in _load_templates():
        th0, tw0 = tpl.shape
        for s in _SCALES:
            tw, th = int(tw0 * s), int(th0 * s)
            if tw >= w or th >= h or tw < 12 or tw > w * 0.35:
                continue
            t2 = cv2.resize(tpl, (tw, th))
            res = cv2.matchTemplate(gray, t2, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= _THRESHOLD)
            for x, y in zip(xs, ys):
                dets.append((float(res[y, x]), int(x), int(y), tw, th))
    dets.sort(reverse=True)
    kept = []
    for sc, x, y, tw, th in dets:
        overlap = False
        for _, kx, ky, ktw, kth in kept:
            if not (x + tw < kx or kx + ktw < x or y + th < ky or ky + kth < y):
                overlap = True
                break
        if not overlap:
            kept.append((sc, x, y, tw, th))
    return [(x, y, tw, th) for _, x, y, tw, th in kept]


def clean_watermark(src_path):
    """Remove GSMArena watermark stamps from an image.

    Returns the path of the cleaned image (cached under video/work/watermark).
    Returns the original path unchanged when OpenCV is unavailable, no
    watermark is detected, or anything fails.
    """
    if not _CV2:
        return src_path
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        key = hashlib.md5(os.path.abspath(src_path).encode()).hexdigest()
        ext = os.path.splitext(src_path)[1].lower() or ".jpg"
        dst = os.path.join(_CACHE_DIR, f"{key}{ext}")
        if os.path.exists(dst) and \
                os.path.getmtime(dst) >= os.path.getmtime(src_path):
            return dst
        img = cv2.imread(src_path)
        if img is None:
            return src_path
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes = _detect(gray)
        if not boxes:
            # cache the negative result too (touch an empty marker is
            # overkill; just return src and re-detect next run)
            return src_path
        h, w = gray.shape
        mask = np.zeros((h, w), np.uint8)
        for x, y, tw, th in boxes:
            x0, y0 = max(0, x - 2), max(0, y - 2)
            x1, y1 = min(w, x + tw + 2), min(h, y + th + 2)
            mask[y0:y1, x0:x1] = 255
        cleaned = cv2.inpaint(img, mask, 4, cv2.INPAINT_NS)
        cv2.imwrite(dst, cleaned)
        return dst
    except Exception:
        return src_path
