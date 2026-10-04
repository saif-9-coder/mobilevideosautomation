"""
Video builder — Techvolution-style spec showcase, rendered with PIL + ffmpeg.
See docs/reference-style.md
"""
import os
import subprocess

from .spec_fields import card_data
from .style_spec import render_card, render_intro, W, H

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output")
WORK_DIR = os.path.join(ROOT, "data", "frames")
FPS = 30


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])


def remove_white_bg(src_path, dst_path, thresh=235):
    """Remove near-white background via border flood-fill (no model download).
    Keeps white areas *inside* the product intact."""
    import numpy as np
    from collections import deque
    from PIL import Image, ImageFilter

    img = Image.open(src_path).convert("RGBA")
    arr = np.array(img)
    h, w, _ = arr.shape
    rgb = arr[..., :3].astype(np.int32)
    # sample background color from the brightest corner pixel (robust to JPEG noise)
    px = img.load()
    corners = [px[0, 0][:3], px[w - 1, 0][:3], px[0, h - 1][:3], px[w - 1, h - 1][:3]]
    bg_color = np.array(max(corners, key=sum), dtype=np.int32)
    dist = np.sqrt(((rgb - bg_color) ** 2).sum(axis=2))
    near_bg = dist < 60

    bg = np.zeros((h, w), dtype=bool)
    dq = deque()
    for x in range(w):
        for y in (0, h - 1):
            if near_bg[y, x]:
                bg[y, x] = True
                dq.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if near_bg[y, x] and not bg[y, x]:
                bg[y, x] = True
                dq.append((y, x))
    while dq:
        y, x = dq.popleft()
        if y > 0 and near_bg[y-1, x] and not bg[y-1, x]:
            bg[y-1, x] = True; dq.append((y-1, x))
        if y < h-1 and near_bg[y+1, x] and not bg[y+1, x]:
            bg[y+1, x] = True; dq.append((y+1, x))
        if x > 0 and near_bg[y, x-1] and not bg[y, x-1]:
            bg[y, x-1] = True; dq.append((y, x-1))
        if x < w-1 and near_bg[y, x+1] and not bg[y, x+1]:
            bg[y, x+1] = True; dq.append((y, x+1))

    arr[bg, 3] = 0
    out = Image.fromarray(arr)
    # feather the cut edge slightly
    alpha = out.split()[3].filter(ImageFilter.GaussianBlur(1))
    out.putalpha(alpha)
    # crop to the non-transparent content (kills leftover margins)
    bbox = out.split()[3].getbbox()
    if bbox:
        out = out.crop(bbox)
    out.save(dst_path)
    return dst_path


def prepare_image(phone):
    """Return path to a background-removed PNG of the phone (or original)."""
    os.makedirs(WORK_DIR, exist_ok=True)
    dst = os.path.join(WORK_DIR, f"phone_{phone['id']}_cut.png")
    if os.path.exists(dst):
        return dst
    src = phone.get("local_image")
    if not src or not os.path.exists(src):
        return None
    # Prefer rembg if its model is already cached (no download at runtime);
    # otherwise use the fast white-background remover (GSMArena shots are on white).
    model_cached = os.path.exists(os.path.expanduser("~/.u2net/u2net.onnx"))
    if model_cached:
        try:
            from rembg import remove
            from PIL import Image
            remove(Image.open(src).convert("RGBA")).save(dst)
            return dst
        except Exception:
            pass
    try:
        return remove_white_bg(src, dst)
    except Exception:
        return src


def _segments_to_mp4(segments, tag):
    """segments: [(png, dur)] -> [(mp4, dur)]"""
    seg_files = []
    for i, (png, dur) in enumerate(segments):
        out = os.path.join(WORK_DIR, f"{tag}_{i}.mp4")
        _run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", png,
              "-t", str(dur), "-r", str(FPS),
              "-vf", f"scale={W}:{H},format=yuv420p",
              "-c:v", "libx264", "-preset", "medium", "-crf", "20", out])
        seg_files.append((out, dur))
    return seg_files


def _xfade_concat(seg_files, final, fade=0.8, music=None):
    """Concatenate segment mp4s with blur-less crossfades. seg_files: [(path, dur)]."""
    if len(seg_files) == 1:
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", seg_files[0][0], "-c", "copy"]
    else:
        cmd = ["ffmpeg", "-y", "-v", "error"]
        for f, _ in seg_files:
            cmd += ["-i", f]
        parts, prev, total = [], "0:v", seg_files[0][1]
        for i in range(1, len(seg_files)):
            off = total - fade
            lab = "vout" if i == len(seg_files) - 1 else f"x{i}"
            parts.append(f"[{prev}][{i}:v]xfade=transition=fade:duration={fade}"
                         f":offset={off}[{lab}]")
            prev, total = lab, total + seg_files[i][1] - fade
        cmd += ["-filter_complex", ";".join(parts), "-map", "[vout]",
                "-r", str(FPS), "-c:v", "libx264", "-preset", "medium",
                "-crf", "20", "-pix_fmt", "yuv420p"]
    if music and os.path.exists(music):
        cmd += ["-i", music, "-c:a", "aac", "-shortest"]
    cmd.append(final)
    _run(cmd)
    return final


def _safe(name):
    return "".join(c if c.isalnum() else "_" for c in name)[:60]


def build_phone_video(phone, style="spec_showcase", secs_per_phone=10,
                      intro=True, music=None):
    """Build the video for one phone dict (from db.get_phone). Returns output path."""
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    data = card_data(phone)
    img_path = prepare_image(phone)
    card_png = os.path.join(WORK_DIR, f"card_{phone['id']}_{style}.png")
    render_card(data, img_path, card_png, theme=style)

    segments = []
    if intro:
        intro_png = os.path.join(WORK_DIR, f"intro_{phone['id']}_{style}.png")
        render_intro(phone["name"].split()[0], data["title"], intro_png,
                     [img_path] if img_path else None, theme=style)
        segments.append((intro_png, 5))
    segments.append((card_png, secs_per_phone))

    seg_files = _segments_to_mp4(segments, f"seg_{phone['id']}")
    final = os.path.join(OUT_DIR, f"{_safe(phone['name'])}_{style}.mp4")
    return _xfade_concat(seg_files, final, music=music)


def build_series_video(phones, brand, series, style="spec_showcase",
                       secs_per_phone=10, music=None, title="EVOLUTION"):
    """One video covering a whole series (phones in chronological order)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    segments = []
    intro_png = os.path.join(WORK_DIR, f"intro_series_{phones[0]['series_id']}_{style}.png")
    imgs = [prepare_image(p) for p in phones[:6]]
    render_intro(brand, f"{series} {title}", intro_png, [i for i in imgs if i],
                 theme=style)
    segments.append((intro_png, 5))

    for ph in phones:
        data = card_data(ph)
        png = os.path.join(WORK_DIR, f"card_{ph['id']}_{style}.png")
        render_card(data, prepare_image(ph), png, theme=style)
        segments.append((png, secs_per_phone))

    seg_files = _segments_to_mp4(segments, "sseg")
    final = os.path.join(OUT_DIR, f"{_safe(brand + '_' + series)}_{style}.mp4")
    return _xfade_concat(seg_files, final, music=music)
