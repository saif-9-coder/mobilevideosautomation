"""
Extracts the exact fields the Techvolution-style card needs from a GSMArena specs dict.
See docs/reference-style.md
"""
import re

MONTHS = {"january": "Jan", "february": "Feb", "march": "Mar", "april": "Apr",
          "may": "May", "june": "Jun", "july": "Jul", "august": "Aug",
          "september": "Sep", "october": "Oct", "november": "Nov", "december": "Dec"}


def _sec(specs, name):
    return specs.get(name, {})


def display(specs):
    d = _sec(specs, "Display")
    size = d.get("Size", "")
    m = re.search(r"([\d.]+)\s*inches", size)
    inches = f'{m.group(1)}"' if m else ""
    panel = d.get("Type", "").split(",")[0].strip()
    if len(panel) > 26:
        panel = " ".join(panel.split()[:3])
    hz = re.search(r"(\d+)\s*Hz", size + " " + d.get("Type", ""))
    return inches, panel, (hz.group(1) + "Hz" if hz else "60Hz")


def camera(specs):
    mc = _sec(specs, "Main Camera")
    val = ""
    for v in mc.values():
        if "MP" in v:
            val = v
            break
    mps = re.findall(r"(\d+)\s*MP", val)
    rear = "/".join(mps) + " MP" if mps else ""
    sc = _sec(specs, "Selfie camera")
    svals = " ".join(sc.values())
    m = re.search(r"(\d+)\s*MP", svals)
    front = f"FRONT {m.group(1)} MP" if m else ""
    return rear, front


def storage_ram(specs):
    internal = _sec(specs, "Memory").get("Internal", "")
    pairs = re.findall(r"(\d+)\s*GB\s*(\d+)\s*GB RAM", internal)
    if pairs:
        st = "/".join(sorted({p[0] for p in pairs}, key=int)) + " GB"
        ram = "/".join(sorted({p[1] for p in pairs}, key=int)) + " GB"
        return st, ram
    nums = re.findall(r"(\d+)\s*GB", internal)
    if len(nums) >= 2:
        return nums[0] + " GB", nums[1] + " GB"
    return "", ""


def battery(specs):
    b = _sec(specs, "Battery").get("Type", "")
    m = re.search(r"(\d+)\s*mAh", b.replace(",", ""))
    if m:
        return f"{int(m.group(1)):,}".replace(",", ".") + " mAh"
    return ""


def weight(specs):
    w = _sec(specs, "Body").get("Weight", "")
    m = re.search(r"~?\s*([\d.]+)\s*g", w)
    return (("~" if w.strip().startswith("~") else "") + m.group(1) + " g") if m else ""


def android_version(specs):
    osv = _sec(specs, "Platform").get("OS", "")
    m = re.search(r"Android\s*([\d.]+)", osv)
    return f"Android {m.group(1)}" if m else ""


def chipset_short(specs):
    c = _sec(specs, "Platform").get("Chipset", "")
    m = re.search(r"Snapdragon\s*([\w ]+?)(?:\s*\(|$)", c)
    if m:
        return "Snapdragon " + m.group(1).strip()
    for brand in ["Exynos", "Dimensity", "Helio", "Tensor", "Kirin", "Apple"]:
        m = re.search(rf"{brand}\s*([\w ]+?)(?:\s*\(|$)", c)
        if m:
            return f"{brand} " + m.group(1).strip()
    return " ".join(c.split()[:3])


def released(specs):
    a = _sec(specs, "Launch").get("Announced", "")
    m = re.search(r"(\d{4}),\s*([A-Za-z]+)", a)
    if m:
        mon = MONTHS.get(m.group(2).lower(), m.group(2)[:3])
        return f"Released {m.group(1)}, {mon}", m.group(1)
    return "", ""


def card_data(phone):
    """phone: dict from db.get_phone. Returns everything the card renderer needs."""
    specs = phone.get("specs", {})
    inches, panel, hz = display(specs)
    rear, front = camera(specs)
    storage, ram = storage_ram(specs)
    rel, year = released(specs)
    return {
        "title": phone["name"],
        "released": rel,
        "year": year,
        "badges": [hz, android_version(specs), chipset_short(specs)],
        "rows": [
            ("display", "DISPLAY", inches, panel),
            ("camera", "CAMERA", rear, front),
            ("storage", "STORAGE", storage, ""),
            ("ram", "RAM", ram, ""),
            ("battery", "BATTERY", battery(specs), ""),
            ("weight", "WEIGHT", weight(specs), ""),
        ],
    }
