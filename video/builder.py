"""
Video builder — renders the spec-showcase style video with ffmpeg.

The exact look (backgrounds, fonts, layout, transitions) is tuned from the
reference video analysis in docs/reference-style.md.
"""
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output")
W, H, FPS = 1920, 1080, 30


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return r


def build_phone_video(phone, style="spec_showcase"):
    """Build the video for one phone dict (from db.get_phone). Returns output path."""
    os.makedirs(OUT_DIR, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in phone["name"])[:60]
    out = os.path.join(OUT_DIR, f"{safe}_{style}.mp4")
    # NOTE: full style implementation lands after the reference-video analysis.
    # Placeholder: title card so the pipeline is end-to-end testable.
    vf = (
        f"color=c=0x0f1420:s={W}x{H}:r={FPS}:d=5,"
        f"drawtext=text='{phone['name']}':fontsize=72:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2"
    )
    _run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", vf,
          "-t", "5", "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p", out])
    return out
