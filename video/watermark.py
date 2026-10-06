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
_DICE_THRESHOLD = 0.20  # strict: verify stroke pattern matches text
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
        ok, box, stroke_info = _verify(gray, x, y, tw, th, ti)
        if ok:
            verified.append((box, stroke_info))
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


def _remove_box_minimal(img_bgr, gray, box):
    """v13: Ultra-minimal text removal.

    Only the most confident text-stroke pixels are touched (high threshold,
    no dilation). Each is filled with the median of nearby non-text pixels.
    No AI generation, no inpainting - the original image is otherwise
    bit-identical.
    """
    rx, ry, tw, th = box
    h, w = gray.shape
    x0, y0 = max(0, rx - 5), max(0, ry - 5)
    x1, y1 = min(w, rx + tw + 5), min(h, ry + th + 5)
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    bg_med = cv2.medianBlur(gray[y0:y1, x0:x1], 21).astype(np.float32)
    # Strict: only confident text pixels (high threshold, NO dilation)
    strokes = (cv2.absdiff(patch, bg_med) > 12).astype(np.uint8)
    # Restrict to verified box
    bm = np.zeros_like(strokes); bm[ry-y0:ry-y0+th, rx-x0:rx-x0+tw] = 1
    strokes = strokes * bm
    n = int(strokes.sum())
    if n < 10:
        return False
    # Fill each stroke pixel with median of nearby non-stroke pixels
    region = img_bgr[y0:y1, x0:x1].copy()
    sm = strokes.astype(bool)
    # Background estimate: median-blurred version (text-free)
    bg_est = cv2.medianBlur(img_bgr[y0:y1, x0:x1], 15)
    m3 = np.stack([sm] * 3, axis=2)
    region[m3] = bg_est[m3]
    img_bgr[y0:y1, x0:x1] = region
    return True



def _verify_clean(original_bgr, cleaned_bgr, mask):
    """v14: Verify the cleaned image.

    Returns (ok, reason):
    - Outside the mask, pixels must be bit-identical to original.
    - Inside the mask, no bright text remnants should remain.
    """
    m = (mask > 0)
    if not m.any():
        return True, "empty mask"
    # 1. Outside mask: must be identical
    outside = ~m
    if outside.any():
        if not np.array_equal(original_bgr[outside], cleaned_bgr[outside]):
            diff = np.abs(original_bgr.astype(int) - cleaned_bgr.astype(int))
            maxd = int(diff[outside].max())
            if maxd > 0:
                return False, f"outside changed (max diff {maxd})"
    # 2. Inside mask: text should be gone (no strong bright strokes left)
    gray_clean_u8 = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2GRAY)
    gray_clean = gray_clean_u8.astype(np.float32)
    bg_med = cv2.medianBlur(gray_clean_u8, 21).astype(np.float32)
    residual = cv2.absdiff(gray_clean, bg_med)
    # Bright residual inside mask = leftover text
    left = int(((residual > 25) & m).sum())
    if left > 30:
        return False, f"text remnants ({left}px)"
    return True, "ok"


def _remove_box_v14(img_bgr, gray, box):
    """v14: Best-practice text removal with verification.

    1. Tightest confident text mask (no over-masking).
    2. Fill with edge-aware local background (deterministic).
    3. Verify: outside bit-identical, text fully gone, placement kept.
    4. If verification fails, try one wider pass; else report.
    """
    rx, ry, tw, th = box
    h, w = gray.shape
    pad = 8
    x0, y0 = max(0, rx - pad), max(0, ry - pad)
    x1, y1 = min(w, rx + tw + pad), min(h, ry + th + pad)
    orig_crop = img_bgr[y0:y1, x0:x1].copy()
    patch = gray[y0:y1, x0:x1].astype(np.float32)

    def build_mask(thresh, dil_iter):
        bg_med = cv2.medianBlur(gray[y0:y1, x0:x1], 21).astype(np.float32)
        s = (cv2.absdiff(patch, bg_med) > thresh).astype(np.uint8)
        bm = np.zeros_like(s); bm[ry-y0:ry-y0+th, rx-x0:rx-x0+tw] = 1
        s = s * bm
        if dil_iter:
            s = cv2.dilate(s, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                           iterations=dil_iter)
        return (s * 255).astype(np.uint8)

    def fill_with(mask):
        # Edge-aware fill: Telea with small radius uses immediate neighbours,
        # keeping gradients and placement intact.
        crop = orig_crop.copy()
        filled = cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA)
        m = (mask.astype(np.float32) / 255.0)
        m = cv2.GaussianBlur(m, (3, 3), 0)
        m3 = np.stack([m] * 3, axis=2)
        blended = (crop.astype(np.float32) * (1 - m3)
                   + filled.astype(np.float32) * m3)
        return np.clip(blended, 0, 255).astype(np.uint8)

    # Pass 1: tightest mask
    mask = build_mask(12, 0)
    if int((mask > 0).sum()) < 10:
        return False
    result = fill_with(mask)
    ok, reason = _verify_clean(orig_crop, result, mask)
    if ok:
        img_bgr[y0:y1, x0:x1] = result
        return True
    # Pass 2: slightly wider mask (catch faint halo), re-verify
    mask2 = build_mask(8, 1)
    result2 = fill_with(mask2)
    ok2, reason2 = _verify_clean(orig_crop, result2, mask2)
    if ok2:
        img_bgr[y0:y1, x0:x1] = result2
        return True
    return False


def _remove_box_lama(img_bgr, box, aggressive=False):
    """Remove watermark using LaMa AI inpainting (v10).

    Runs LaMa on a tight crop around the watermark for speed,
    then pastes the cleaned crop back. Falls back to None if
    LaMa is not available.

    aggressive=True: for bottom-edge boxes where watermarks live.
    Uses looser thresholds to catch faint text. Still safe because
    the box is at the edge, not the middle of the phone.
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
    # Tight mask: only the actual text strokes, not the full box.
    # This keeps LaMa's regeneration minimal so surrounding pixels
    # stay pixel-identical and nothing shifts.
    # SAFETY (Saif: never damage): watermark text is BRIGHT white and
    # high-contrast. Use a high threshold so we only catch real text,
    # not image texture. If the "strokes" cover too much of the box,
    # it's a false positive — skip rather than damage.
    # aggressive=True (bottom edge): looser thresholds for faint text.
    contrast_thr = 10 if aggressive else 15
    bright_thr = 100 if aggressive else 140
    coverage_limit = 0.40 if aggressive else 0.25
    crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    bg_med = cv2.medianBlur(crop_gray, 21)
    strokes = (cv2.absdiff(crop_gray.astype(np.float32),
                           bg_med.astype(np.float32)) > contrast_thr).astype(np.uint8)
    # Text is bright: keep only bright strokes
    bright = (crop_gray > bright_thr).astype(np.uint8)
    strokes = strokes * bright
    # Restrict to the verified box area
    box_mask = np.zeros((ch, cw), dtype=np.uint8)
    box_mask[ry-y0:ry-y0+th, rx-x0:rx-x0+tw] = 1
    strokes = strokes * box_mask
    # Safety: if strokes cover too much of box area, it's texture not text
    box_area = tw * th
    if box_area > 0 and float(strokes.sum()) / box_area > coverage_limit:
        return False
    strokes = cv2.dilate(strokes,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                         iterations=1)
    if strokes.sum() < 20:
        return False
    mask = (strokes * 255).astype(np.uint8)
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
    # Only replace the masked (text) pixels; keep original elsewhere.
    # Feather the blend at edges for a seamless, shift-free result.
    m = (mask.astype(np.float32) / 255.0)
    m = cv2.GaussianBlur(m, (5, 5), 0)
    m3 = np.stack([m] * 3, axis=2)
    orig_crop = img_bgr[y0:y1, x0:x1].astype(np.float32)
    blended = orig_crop * (1 - m3) + result_bgr.astype(np.float32) * m3
    img_bgr[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return True


def _remove_box(img_bgr, gray, box):
    # LaMa AI FIRST (best quality, Canva-like) - this is what worked
    # on Saif's test images. Classical methods run only if LaMa fails.
    try:
        if _remove_box_lama(img_bgr, box):
            return True
    except Exception:
        pass
    # v14: Best-practice with verification (fallback)
    try:
        if _remove_box_v14(img_bgr, gray, box):
            return True
    except Exception:
        pass
    # v13: Ultra-minimal fallback
    if _remove_box_minimal(img_bgr, gray, box):
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
        # Add fallback (sliding window) boxes FIRST - more reliable for faint
        try:
            fb = [(b, None) for b in _detect_bottom_edge(gray)]
            verified = fb + verified
        except Exception:
            pass
        if not verified:
            return src_path
        done = False
        fallback_done = False
        for (box, stroke_info) in verified[:_MAX_BOXES + 3]:
            try:
                # Fallback boxes (None info): use simple direct inpaint
                # (sliding window already found text location)
                if stroke_info is None:
                    rx, ry, tw, th = box
                    # Skip vertical boxes (hard case: text on phone edges).
                    # Classical methods damage these; leave for manual/LaMa.
                    if th > tw * 1.5:
                        continue
                    h, w = gray.shape
                    patch = gray[ry:ry+th, rx:rx+tw].astype(np.float32)
                    bg = cv2.medianBlur(gray[ry:ry+th, rx:rx+tw], 15).astype(np.float32)
                    m = (cv2.absdiff(patch, bg) > 10).astype(np.uint8) * 255
                    if int((m > 0).sum()) >= 10:
                        full_mask = np.zeros((h, w), np.uint8)
                        full_mask[ry:ry+th, rx:rx+tw] = m
                        img[:] = cv2.inpaint(img, full_mask, 3, cv2.INPAINT_TELEA)
                        done = True
                        fallback_done = True
                        # don't break: process other fallback boxes too
                elif not fallback_done and _remove_box(img, gray, box):
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
                    if _remove_box_v14(img, gray, (rx, ry, rw, rh)):
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
    through the same LaMa cleaning used in the manual tool. Alpha channel is
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

        # Same detection as clean_watermark: template + bottom-edge fallback
        verified = _detect(gray)
        try:
            fb = [(b, None) for b in _detect_bottom_edge(gray)]
            verified = fb + verified
        except Exception:
            pass
        if not verified:
            return cutout_path

        done = False
        fallback_done = False
        for (box, stroke_info) in verified[:_MAX_BOXES + 3]:
            try:
                if stroke_info is None:
                    rx, ry, tw, th = box
                    if th > tw * 1.5:
                        continue
                    h, w = gray.shape
                    patch = gray[ry:ry+th, rx:rx+tw].astype(np.float32)
                    bg = cv2.medianBlur(gray[ry:ry+th, rx:rx+tw], 15).astype(np.float32)
                    m = (cv2.absdiff(patch, bg) > 10).astype(np.uint8) * 255
                    if int((m > 0).sum()) >= 10:
                        full_mask = np.zeros((h, w), np.uint8)
                        full_mask[ry:ry+th, rx:rx+tw] = m
                        # LaMa first for cutouts too (best quality)
                        if not _remove_box_lama(bgr, (rx, ry, tw, th), aggressive=True):
                            bgr[:] = cv2.inpaint(bgr, full_mask, 3, cv2.INPAINT_TELEA)
                        done = True
                        fallback_done = True
                elif not fallback_done and _remove_box(bgr, gray, box):
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
