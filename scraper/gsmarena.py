"""
GSMArena scraper: brands -> phones -> full specifications.
Stores everything in SQLite (see db.py).
"""
import re
import time
import requests
from urllib.parse import urljoin

BASE = "https://www.gsmarena.com/"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}


def get(url, retries=3, delay=1.0):
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200 and len(r.text) > 1000:
                return r.text
        except Exception:
            pass
        time.sleep(delay * (i + 1))
    return None


def scrape_brands():
    """Return [{'name': ..., 'url': ...}] for every brand on makers.php."""
    html = get(urljoin(BASE, "makers.php3"))
    if not html:
        return []
    brands = []
    for m in re.finditer(r'<a href="([a-z0-9_\-]+-phones-\d+\.php)">([^<]+)</a>', html):
        url, name = m.group(1), m.group(2).strip()
        brands.append({"name": name, "url": urljoin(BASE, url)})
    # dedupe, keep order
    seen, out = set(), []
    for b in brands:
        if b["url"] not in seen:
            seen.add(b["url"])
            out.append(b)
    return out


def scrape_brand_phones(brand_url):
    """Return [{'name': ..., 'url': ..., 'img': ...}] for all phones of a brand (all pages)."""
    phones, page = [], brand_url
    while page:
        html = get(page)
        if not html:
            break
        for m in re.finditer(
            r'<li>\s*<a href="([^"]+\.php)"><img src=([^\s>]+)[^>]*>\s*<strong><span>([^<]+)</span>',
            html,
        ):
            url, img, name = m.group(1), m.group(2), m.group(3).strip()
            phones.append({"name": name, "url": urljoin(BASE, url), "img": img})
        nxt = re.search(r'<a class="pages-nav"[^>]*href="([^"]+)"[^>]*>\s*<i[^>]*>\s*&#8250;', html)
        # fallback: look for next-page link pattern
        if not nxt:
            nxt = re.search(r'href="([^"]*p\d+\.php)"[^>]*title="Next page"', html)
        page = urljoin(BASE, nxt.group(1)) if nxt else None
        time.sleep(0.6)
    # dedupe
    seen, out = set(), []
    for p in phones:
        if p["url"] not in seen:
            seen.add(p["url"])
            out.append(p)
    return out


def _clean(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def scrape_phone_specs(phone_url):
    """Return dict with name, image and a specs dict {section: {field: value}}."""
    html = get(phone_url)
    if not html:
        return None
    data = {"url": phone_url, "specs": {}}

    m = re.search(r'<h1 class="specs-phone-name-title"[^>]*>([^<]+)</h1>', html)
    data["name"] = m.group(1).strip() if m else ""

    m = re.search(r'<div class="specs-photo-main">.*?<img[^>]*src=([^\s>]+)', html, re.S)
    data["image_url"] = m.group(1).strip('"') if m else ""

    # quick key specs shown under the title
    quick = {}
    for mm in re.finditer(
        r'<span class="specs-brief-accent">.*?</span>\s*<span[^>]*>(.*?)</span>', html, re.S
    ):
        pass  # brief spans are icon-led; full table below is authoritative

    # full spec table
    for t in re.finditer(r'<table[^>]*>(.*?)</table>', html, re.S):
        body = t.group(1)
        th = re.search(r'<th[^>]*>(.*?)</th>', body, re.S)
        section = _clean(th.group(1)) if th else "General"
        fields = {}
        for row in re.finditer(
            r'<td class="ttl">.*?<a[^>]*>([^<]+)</a>.*?</td>\s*<td class="nfo"[^>]*>(.*?)</td>',
            body, re.S,
        ):
            fields[row.group(1).strip()] = _clean(row.group(2))
        if fields:
            data["specs"][section] = fields

    # flatten the most-used fields for convenience
    flat = {}
    for section, fields in data["specs"].items():
        for k, v in fields.items():
            flat[f"{section}::{k}"] = v
    data["flat"] = flat
    return data


if __name__ == "__main__":
    brands = scrape_brands()
    print(f"brands: {len(brands)}")
    for b in brands[:5]:
        print(" -", b["name"], b["url"])
