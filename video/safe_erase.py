"""Safe text eraser — the ONLY eraser used by MobileVideosAutomation.

Saif's rule: the phone image must never be damaged, not even a little.
Only the watermark text goes; everything else stays bit-identical.

How it works (no AI, fully deterministic, two passes):
1. ROBUST BG ESTIMATE: two-stage median (cap bright outliers first).
   Plain median is polluted by nearby bright phone edges, hiding faint
   text. The background behind the text is what Saif asked to fill with
   ("text ko wohi color fill kr dy jo osky pechy ha").
2. CANDIDATES: pixels brighter than the background estimate, in the
   region. Watermark text is semi-transparent white — always brighter.
3. TEXT STRUCTURE FILTER: keep only candidates in horizontal runs of
   3+ px. Watermark text is horizontal; noise specks are isolated and
   phone edges are vertical (short horizontal runs).
4. EDGE PROTECTION: pixels with very strong gradients (phone trim) are
   never touched.
5. Fill kept pixels with the background color. A second pass catches
   faint leftovers.
6. ONLY filled pixels change. Everything else is bit-identical by
   construction.

erase_text_region(img_bgr, region_mask, min_coverage=0.5) -> (out, info)
min_coverage: auto pipeline passes 0.5 (box must be mostly text);
dashboard passes 0.0 (user explicitly brushed the watermark).
"""

try:
    import cv2
    import numpy as np
    _CV2 = True
except Exception:
    _CV2 = False
    cv2 = None
    np = None

_BG_KERNEL = 61
_BRIGHT_THR = 2       # candidate if brighter than bg by this much
_RUN_LEN = 3          # min horizontal run length for text
_EDGE_RUN = 15        # vertical runs longer than this = phone edge
_MIN_CAND = 15


def _bg_estimate(channel):
    """Robust background estimate for bright-text removal.

    Morphological opening (25px) removes bright text strokes while
    preserving the background. Plain median is polluted by nearby bright
    phone edges, hiding faint text.
    """
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    return cv2.morphologyEx(channel, cv2.MORPH_OPEN, k)


def _erase_pass(img_bgr, region):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg_gray = _bg_estimate(gray)
    bg_col = _bg_estimate(img_bgr)
    # Candidates: brighter than bg, in region.
    cand = (((gray.astype(np.int16) > bg_gray.astype(np.int16) + _BRIGHT_THR)
             .astype(np.uint8)) & region)
    # Vertical runs: phone edges are tall vertical structures; text is not.
    # Exclude pixels in vertical runs longer than _EDGE_RUN.
    vlen = np.zeros_like(cand, dtype=np.int32)
    h, w = cand.shape
    for x in range(w):
        col = cand[:, x]
        if not col.any():
            continue
        d = np.diff(np.concatenate([[0], col, [0]]))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            if e - s > _EDGE_RUN:
                vlen[s:e, x] = 1
    cand = cand & (1 - vlen.astype(np.uint8))
    # Text structure: horizontal runs.
    mask = np.zeros_like(cand)
    for y in range(h):
        row = cand[y]
        if not row.any():
            continue
        d = np.diff(np.concatenate([[0], row, [0]]))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            if e - s >= _RUN_LEN:
                mask[y, s:e] = 1
    out = img_bgr.copy()
    sel = mask > 0
    out[sel] = bg_col[sel]
    return out, mask, int(sel.sum())


def erase_text_region(img_bgr, region_mask, min_coverage=0.5, prior_mask=None):
    """Erase watermark text inside region_mask. See module docstring."""
    if not _CV2:
        return img_bgr, {"ok": False, "reason": "no cv2"}
    region = (np.asarray(region_mask) > 0).astype(np.uint8)
    if not region.any():
        return img_bgr, {"ok": False, "reason": "empty region"}
    info_base = {"replaced_px": 0, "outside_identical": True,
                 "remnants_px": 0, "ok": False}

    out1, mask1, n1 = _erase_pass(img_bgr, region)
    if n1 < _MIN_CAND:
        info_base["reason"] = "no text found"
        return img_bgr, info_base
    if min_coverage > 0:
        ys, xs = (mask1 > 0).nonzero()
        span = (xs.max() - xs.min() + 1) if len(xs) else 0
        ryn, rxn = region.nonzero()
        region_w = (rxn.max() - rxn.min() + 1) if len(rxn) else 1
        if n1 < 100 or span < min_coverage * region_w:
            info_base["reason"] = "box rejected"
            return img_bgr, info_base

    out2, mask2, n2 = _erase_pass(out1, region)
    changed = ((mask1 > 0) | (mask2 > 0))
    outside = ~changed
    info = {
        "replaced_px": n1 + n2,
        "outside_identical": bool(
            np.array_equal(img_bgr[outside], out2[outside])),
        "remnants_px": 0,
        "ok": False,
    }
    g2 = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)
    b2 = _bg_estimate(g2)
    rem = ((cv2.absdiff(g2, b2) > 25) & (g2 > 140)
           & (region > 0) & ~changed).sum()
    info["remnants_px"] = int(rem)
    info["ok"] = bool(info["outside_identical"] and info["remnants_px"] <= 10)
    return out2, info
