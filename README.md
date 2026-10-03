# 📱 MobileVideosAutomation

Automated pipeline: **GSMArena → Database → Production-ready YouTube videos** + local dashboard.

## What it does

1. **Scraper** (`scraper/`) — crawls GSMArena: all brands → every phone → full spec tables → phone images. Stores in SQLite with separate IDs per brand, series (e.g. Galaxy S, iPhone) and phone.
2. **Video builder** (`video/`) — renders 1080p videos from the scraped data with ffmpeg (style defined in `docs/reference-style.md`).
3. **Dashboard** (`dashboard/`) — local web UI to browse brands/series/phones, view specs & images, queue video builds and watch results.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Scrape brands
python scraper/run_scraper.py brands

# 2. Scrape phone lists (one brand or all)
python scraper/run_scraper.py phones --brand Samsung
python scraper/run_scraper.py phones --all

# 3. Scrape full specifications
python scraper/run_scraper.py specs --brand Samsung
python scraper/run_scraper.py specs --all

# 4. Download phone images
python scraper/run_scraper.py images --brand Samsung

# 5. Build a video
python video/run_builder.py --phone-id 123 --style spec_showcase
python video/run_builder.py --series-id 5 --style spec_showcase

# 6. Dashboard (http://127.0.0.1:5000)
python dashboard/app.py
```

## Automation (cron)

Keep data fresh — add to crontab (`crontab -e`):

```cron
# every night at 3 AM: refresh brands, phone lists and specs
0 3 * * * cd /path/to/mobilevideosautomation && python scraper/run_scraper.py brands >> data/scraper.log 2>&1
30 3 * * * cd /path/to/mobilevideosautomation && python scraper/run_scraper.py phones --all >> data/scraper.log 2>&1
0 5 * * * cd /path/to/mobilevideosautomation && python scraper/run_scraper.py specs --all >> data/scraper.log 2>&1
```

The exact commands to run are also in `scraper/run_scraper.py --help` header.

## Database

SQLite at `data/mva.db` (override with `MVA_DB` env var). Tables: `brands`, `series`, `phones`, `videos`.

## Notes

- Phone images from GSMArena have white backgrounds; use `rembg` (in requirements) to remove backgrounds before video assembly if the style needs transparent cutouts.
- Be polite to GSMArena: the scraper already sleeps between requests. Don't hammer it.
