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
_CACHE_VERSION = "v10"
_TEMPLATES = ("wm_template.png", "wm_template_b.png", "wm_template_c.png")
_SCALES = (0.6, 0.75, 0.9, 1.05, 1.2, 1.4)
_NCC_THRESHOLD = 0.78
_DICE_THRESHOLD = 0.15
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
    # Dice between image strokes and template strokes at refined position
    x0, y0 = max(0, rx - 3), max(0, ry - 3)
    x1, y1 = min(w, rx + tw + 3), min(h, ry + th + 3)
    patch = gray[y0:y1, x0:x1]
    img_st = _image_strokes(patch)
    ts = cv2.resize(tpl_strokes[tpl_idx], (x1 - x0, y1 - y0))
    inter = np.logical_and(img_st, ts).sum()
    dice = inter / ts.sum() if ts.sum() else 0
    if dice < _DICE_THRESHOLD:
        return False, None, None
    return True, (rx, ry, tw, th), (img_st, (x0, y0))


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
        ok, box, stroke_info = _verify(gray, x, y, tw, th, ti)
        if ok:
            verified.append((box, stroke_info))
    return verified




# LaMa AI inpainting (professional quality). Loaded lazily.
_LAMA = None
def _get_lama():
    global _LAMA
    if _LAMA is None:
        try:
            from simple_lama_inpainting import SimpleLama
            _LAMA = SimpleLama()
        except Exception:
            _LAMA = False
    return _LAMA if _LAMA is not False else None

def _remove_box_lama(img_bgr, box):
    """Remove watermark using LaMa AI inpainting (v10).

    Runs LaMa on a tight crop around the watermark for speed,
    then pastes the cleaned crop back. Falls back to None if
    LaMa is not available.
    """
    lama = _get_lama()
    if lama is None:
        return False
    rx, ry, tw, th = box
    h, w = img_bgr.shape[:2]
    # Crop with padding for context (LaMa needs surrounding area)
    pad = 40
    x0, y0 = max(0, rx - pad), max(0, ry - pad)
    x1, y1 = min(w, rx + tw + pad), min(h, ry + th + pad)
    crop_bgr = img_bgr[y0:y1, x0:x1]
    ch, cw = crop_bgr.shape[:2]
    # Mask for the watermark box within the crop
    mask = np.zeros((ch, cw), dtype=np.uint8)
    mask[ry-y0:ry-y0+th, rx-x0:rx-x0+tw] = 255
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                      iterations=2)
    # LaMa expects RGB
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    try:
        result_pil = lama(crop_rgb, mask)
    except Exception:
        return False
    result_rgb = np.array(result_pil)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    # LaMa may pad output; resize back to crop size
    if result_bgr.shape[:2] != (y1 - y0, x1 - x0):
        result_bgr = cv2.resize(result_bgr, (x1 - x0, y1 - y0))
    img_bgr[y0:y1, x0:x1] = result_bgr
    return True


def _remove_box(img_bgr, gray, box):
    # v10: Try LaMa AI first (professional quality)
    if _remove_box_lama(img_bgr, box):
        return True
    # Fallback to Telea inpainting
    """Erase verified watermark text with tight-mask Telea inpainting.

    Only the exact text-stroke pixels are interpolated (from immediate
    neighbours); everything else stays bit-identical, so texture, grain
    and gradients are preserved like professional tools.
    """
    rx, ry, tw, th = box
    h, w = gray.shape
    x0, y0 = max(0, rx - 3), max(0, ry - 3)
    x1, y1 = min(w, rx + tw + 3), min(h, ry + th + 3)
    if x1 - x0 < 8 or y1 - y0 < 4:
        return False
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    bg_med = cv2.medianBlur(gray[y0:y1, x0:x1], 21).astype(np.float32)
    # Aggressive but tight stroke mask: the box is already verified to
    # contain the watermark, so catch faint halo pixels too. Dilated
    # slightly to cover anti-aliased text edges fully.
    strokes = (cv2.absdiff(patch, bg_med) > 4).astype(np.uint8)
    strokes = cv2.dilate(strokes,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                         iterations=2)
    if strokes.sum() < 20:
        return False
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = strokes * 255
    img_bgr[:] = cv2.inpaint(img_bgr, mask, 4, cv2.INPAINT_TELEA)
    return True


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
        verified = _detect(gray)
        if not verified:
            return src_path
        done = False
        for (box, _stroke_info) in verified[:_MAX_BOXES]:
            try:
                if _remove_box(img, gray, box):
                    done = True
            except Exception:
                continue
        if not done:
            return src_path
        cv2.imwrite(dst, img)
        return dst
    except Exception:
        return src_path
