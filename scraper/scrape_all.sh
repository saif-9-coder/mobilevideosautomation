#!/bin/bash
# Full scrape pipeline: phone lists -> specs -> HQ images, all brands.
# Safe to re-run: every stage skips already-scraped rows.
cd "$(dirname "$0")/.." || exit 1
LOG="data/scrape_all.log"
echo "=== pipeline started $(date -u) ===" >> "$LOG"
echo "[1/3] phone lists for all brands..." | tee -a "$LOG"
timeout 1800 python3 scraper/run_scraper.py phones --all >> "$LOG" 2>&1
echo "[2/3] specs for all phones (missing only)..." | tee -a "$LOG"
timeout 20000 python3 scraper/run_scraper.py specs --all --missing-only >> "$LOG" 2>&1
echo "[3/3] HQ images for all phones (missing only)..." | tee -a "$LOG"
timeout 20000 python3 scraper/run_scraper.py images >> "$LOG" 2>&1
echo "=== pipeline finished $(date -u) ===" | tee -a "$LOG"
