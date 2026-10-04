"""
Renders the Techvolution-style phone card as a PNG with PIL.
See docs/reference-style.md
"""
import math
import os

from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1920, 1080


def set_quality(quality):
    """'1080p' (1920x1080) or '1440p' (2560x1440). Must be called before rendering."""
    global W, H
    if quality == "1440p":
        W, H = 2560, 1440
    else:
        W, H = 1920, 1080

THEMES = {
    "spec_showcase": {  # Techvolution light style (default)
        "bg": (199, 208, 222), "grid": (178, 189, 208),
        "text": (20, 22, 28), "gray": (96, 106, 124), "icon": (107, 118, 136),
        "accent": (255, 140, 0), "year_alpha": 46,
    },
    "dark_pro": {  # dark navy + electric blue
        "bg": (13, 18, 32), "grid": (28, 38, 62),
        "text": (240, 244, 252), "gray": (148, 160, 182), "icon": (110, 168, 255),
        "accent": (255, 140, 0), "year_alpha": 40,
    },
    "cream_minimal": {  # warm minimal
        "bg": (246, 241, 232), "grid": (228, 220, 203),
        "text": (32, 30, 26), "gray": (130, 120, 104), "icon": (176, 148, 96),
        "accent": (214, 96, 32), "year_alpha": 44,
    },
}
TEMPLATE_CHOICES = [
    ("spec_showcase", "Spec Showcase (light)"),
    ("dark_pro", "Dark Pro"),
    ("cream_minimal", "Cream Minimal"),
]

BG = THEMES["spec_showcase"]["bg"]
GRID = THEMES["spec_showcase"]["grid"]
BLACK = THEMES["spec_showcase"]["text"]
GRAY = THEMES["spec_showcase"]["gray"]
ICON = THEMES["spec_showcase"]["icon"]

FONT_BOLD = os.environ.get("MVA_FONT_BOLD", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
FONT_REG = os.environ.get("MVA_FONT_REG", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _th(theme):
    return THEMES.get(theme, THEMES["spec_showcase"])


def background(theme="spec_showcase"):
    t = _th(theme)
    img = Image.new("RGB", (W, H), t["bg"])
    d = ImageDraw.Draw(img)
    # subtle wavy grid
    for y in range(0, H + 90, 90):
        pts = [(x, y + 14 * math.sin(x / 260 + y / 300)) for x in range(0, W + 20, 20)]
        d.line(pts, fill=t["grid"], width=2)
    for x in range(0, W + 140, 140):
        pts = [(x + 14 * math.sin(y / 260 + x / 300), y) for y in range(0, H + 20, 20)]
        d.line(pts, fill=t["grid"], width=2)
    return img


def draw_icon(d, kind, x, y, s=64, color=ICON, w=5):
    """Simple gray line icons."""
    if kind == "display":
        d.rounded_rectangle([x, y, x + s, y + s], radius=8, outline=color, width=w)
        d.line([x + 10, y + s - 10, x + s - 10, y + s - 10], fill=color, width=w)
    elif kind == "camera":
        d.rounded_rectangle([x, y + 12, x + s, y + s - 6], radius=10, outline=color, width=w)
        d.ellipse([x + s / 2 - 13, y + s / 2 - 7, x + s / 2 + 13, y + s / 2 + 19],
                  outline=color, width=w)
        d.line([x + 14, y + 12, x + 24, y + 2, x + s - 14, y + 2, x + s - 8, y + 12],
               fill=color, width=w)
    elif kind == "storage":
        d.polygon([(x + 14, y), (x + s, y), (x + s, y + s), (x, y + s), (x, y + 14)],
                  outline=color, width=w)
        d.line([x + 14, y + 22, x + s - 12, y + 22], fill=color, width=4)
        d.line([x + 14, y + 34, x + s - 12, y + 34], fill=color, width=4)
    elif kind == "ram":
        d.rectangle([x + 12, y + 12, x + s - 12, y + s - 12], outline=color, width=w)
        for px in (x + 22, x + 32, x + 42):
            d.line([px, y, px, y + 12], fill=color, width=4)
            d.line([px, y + s - 12, px, y + s], fill=color, width=4)
    elif kind == "battery":
        d.rounded_rectangle([x, y + 14, x + s - 12, y + s - 14], radius=6,
                            outline=color, width=w)
        d.rectangle([x + s - 12, y + s / 2 - 8, x + s - 2, y + s / 2 + 8], fill=color)
        d.rectangle([x + 10, y + 24, x + 30, y + s - 24], fill=color)
    elif kind == "weight":
        d.ellipse([x + 8, y + 14, x + s - 8, y + s], outline=color, width=w)
        d.arc([x + s / 2 - 12, y - 6, x + s / 2 + 12, y + 18], 180, 360, fill=color, width=w)


def drop_shadow(base, img, pos, blur=28, offset=(20, 26), opacity=110):
    sh = Image.new("RGBA", base.size, (0, 0, 0, 0))
    alpha = img.split()[-1] if img.mode == "RGBA" else None
    black = Image.new("RGBA", img.size, (0, 0, 0, opacity))
    if alpha:
        black.putalpha(alpha.point(lambda a: a * opacity // 255))
    sh.alpha_composite(black, (pos[0] + offset[0], pos[1] + offset[1]))
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    base.alpha_composite(sh)
    return base


def render_card(data, phone_imgs=None, out_path=None, theme="spec_showcase"):
    """
    data: dict from spec_fields.card_data — title, released, year, badges, rows.
    phone_imgs: single path (back-compat) or list of paths (all angles, overlapping).
    Returns PIL image (also saved to out_path if given).
    """
    t = _th(theme)
    BLACK, GRAY, ICON = t["text"], t["gray"], t["icon"]
    u = W / 1920
    S = lambda v: max(1, int(v * u))
    img = background(theme).convert("RGBA")
    if (img.width, img.height) != (W, H):
        img = img.resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    fb = lambda s: font(FONT_BOLD, S(s))
    fr = lambda s: font(FONT_REG, S(s))

    # year marker (behind everything else)
    if data.get("year"):
        yf = font(FONT_BOLD, S(300))
        tw = d.textlength(data["year"], font=yf)
        timg = Image.new("RGBA", (int(tw) + S(40), S(360)), (0, 0, 0, 0))
        ImageDraw.Draw(timg).text((S(20), 0), data["year"], font=yf,
                                  fill=BLACK + (t["year_alpha"],))
        img.alpha_composite(timg, (W - int(tw) - S(420), H - S(380)))

    # title + released (auto-fit title so it never hits the badges)
    tsize = 96
    while tsize > 48:
        tf = fb(tsize)
        if d.textlength(data["title"], font=tf) < 950 * u:
            break
        tsize -= 6
    d.text((S(90), S(60)), data["title"], font=fb(tsize), fill=BLACK)
    if data.get("released"):
        d.text((S(94), S(178)), data["released"], font=fr(40), fill=GRAY)

    # top-right badges
    bx = W - S(90)
    for badge in reversed([b for b in data.get("badges", []) if b]):
        bw = d.textlength(badge, font=fb(30)) + S(56)
        bx -= bw
        d.rounded_rectangle([bx, S(84), bx + bw, S(148)], radius=S(32),
                            outline=GRAY, width=S(3))
        d.text((bx + S(28), S(96)), badge, font=fb(30), fill=BLACK)
        bx -= S(24)

    # spec rows
    y = S(280)
    for kind, label, value, sub in data["rows"]:
        if not value and not sub:
            continue
        draw_icon(d, kind, S(90), y + S(8), s=S(64), color=ICON, w=S(5))
        d.text((S(185), y), label, font=fb(30), fill=GRAY)
        if value:
            d.text((S(185), y + S(38)), value, font=fb(58), fill=BLACK)
        if sub:
            d.text((S(185), y + S(104)), sub, font=fb(34), fill=BLACK)
        y += S(140) if sub else S(112)

    # phone image, right side — single hero shot (front+back composite)
    if isinstance(phone_imgs, str):
        phone_imgs = [phone_imgs]
    imgs = [pp for pp in (phone_imgs or []) if pp and os.path.exists(pp)]
    if imgs:
        ph = Image.open(imgs[0]).convert("RGBA")
        th = S(800)
        tw = int(ph.width * th / ph.height)
        # keep the image clear of the spec text (max width)
        max_w = W - S(1050)
        if tw > max_w:
            tw = max_w
            th = int(ph.height * tw / ph.width)
        ph = ph.resize((tw, th), Image.LANCZOS)
        px, py = W - S(90) - tw, (H - th) // 2 + S(20)
        img = drop_shadow(img, ph, (px, py), blur=S(28),
                          offset=(S(20), S(26)), opacity=110)
        img.alpha_composite(ph, (px, py))

    out = img.convert("RGB")
    if out_path:
        out.save(out_path, quality=95)
    return out


def render_intro(brand, series, out_path, phone_imgs=None, theme="spec_showcase"):
    """Thumbnail-style intro card: BRAND / SERIES / EVOLUTION."""
    t = _th(theme)
    BLACK, GRAY, ACCENT = t["text"], t["gray"], t["accent"]
    u = W / 1920
    S = lambda v: max(1, int(v * u))
    img = background(theme).convert("RGBA")
    if (img.width, img.height) != (W, H):
        img = img.resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    fb = lambda s: font(FONT_BOLD, S(s))
    d.text((S(90), S(200)), brand.upper(), font=fb(120), fill=BLACK)
    d.text((S(94), S(350)), f"{series.upper()} EVOLUTION", font=fb(90), fill=ACCENT)
    d.text((S(96), S(480)), "Full specifications showcase",
           font=font(FONT_REG, S(44)), fill=GRAY)
    if phone_imgs:
        if isinstance(phone_imgs, str):
            phone_imgs = [phone_imgs]
        x = S(90)
        for pp in phone_imgs[:8]:
            if pp and os.path.exists(pp):
                ph = Image.open(pp).convert("RGBA")
                th = S(300)
                tw = int(ph.width * th / ph.height)
                img.alpha_composite(ph.resize((tw, th), Image.LANCZOS), (x, S(700)))
                x += tw + S(30)
    out = img.convert("RGB")
    out.save(out_path, quality=95)
    return out
