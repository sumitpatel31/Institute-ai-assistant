"""
website_scraper.py — Scrapes NareshIT course schedule page.

Fetches the HTML from the configured URL, locates all <table> elements,
parses them with BeautifulSoup, converts to clean DataFrames, and saves
both structured JSON and readable text.

Called automatically at app startup and whenever the user clicks
"Refresh Website Data".
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import settings
from text_cleaner import TextCleaner
from utils import ensure_dir, logger


class WebsiteScraper:
    """Scrape structured table data from the NareshIT website."""

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def __init__(self, url: str | None = None) -> None:
        self.url = url or settings.website_url
        self.cleaner = TextCleaner()
        self.tables_data: list[dict[str, Any]] = []
        self.cleaned_text: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def scrape_and_save(self) -> str:
        """
        Full pipeline: fetch → parse tables → clean → save.
        Returns the cleaned text that should be indexed into the vector store.
        """
        logger.info("Starting website scrape for: %s", self.url)
        html = self._fetch_html()
        if html is None:
            logger.warning("No HTML fetched — returning previously cached data if any.")
            return self._load_cached_text()

        all_records = self._extract_tables(html)
        if not all_records:
            logger.warning("No table data extracted from the page.")
            return ""

        # Deduplicate
        df = pd.DataFrame(all_records)
        df = df.drop_duplicates().reset_index(drop=True)

        # Convert to readable text
        self.cleaned_text = self._dataframe_to_text(df)
        self.cleaned_text = self.cleaner.clean_text(self.cleaned_text)

        # Save JSON + TXT
        self._save_results(df)

        logger.info(
            "Website scrape complete — %d unique rows, %d chars of text.",
            len(df),
            len(self.cleaned_text),
        )
        return self.cleaned_text

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _fetch_html(self) -> str | None:
        """GET the page HTML.  Returns None on failure."""
        try:
            resp = requests.get(self.url, headers=self.HEADERS, timeout=30)
            resp.raise_for_status()
            logger.info("Fetched %d bytes from %s", len(resp.text), self.url)
            return resp.text
        except requests.RequestException as exc:
            logger.error("Failed to fetch website: %s", exc)
            return None

    def _extract_tables(self, html: str) -> list[dict[str, str]]:
        """
        Find every <table> in *html*, extract header→row mappings.
        Only keeps rows that have at least one non-empty cell.
        """
        soup = BeautifulSoup(html, "lxml")
        tables = soup.find_all("table")
        logger.info("Found %d table(s) on the page.", len(tables))

        all_records: list[dict[str, str]] = []

        for table_idx, table in enumerate(tables):
            thead = table.find("thead")
            if not thead:
                logger.debug("Table %d has no <thead> — skipping.", table_idx)
                continue

            # Column names from <th> elements
            headers: list[str] = []
            for th in thead.find_all("th"):
                # Get clean text, strip HTML tags inside
                raw = th.get_text(separator=" ", strip=True)
                # Normalise known messy patterns
                raw = re.sub(r"\s+", " ", raw).strip()
                # Remove "activate to sort …" artefacts
                raw = re.sub(r"activate to sort.*", "", raw, flags=re.IGNORECASE).strip()
                if raw:
                    headers.append(raw)

            if not headers:
                continue

            tbody = table.find("tbody") or table
            for tr in tbody.find_all("tr"):
                cells = tr.find_all(["td", "th"])
                row: dict[str, str] = {}
                for col_idx, cell in enumerate(cells):
                    if col_idx >= len(headers):
                        break
                    val = cell.get_text(separator=" ", strip=True)
                    val = re.sub(r"\s+", " ", val).strip()
                    # Skip if cell has only a link text like "Register Now"
                    if val and val.lower() not in ("register now",):
                        row[headers[col_idx]] = val

                if row:
                    row["_source_table"] = f"Website → Course Schedule"
                    all_records.append(row)

        return all_records

    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        """
        Convert DataFrame rows into natural-language sentences suitable
        for embedding and retrieval.

        Example output:
            Course: Full Stack JAVA Placement Program, Faculty: Team Of Experts,
            Date: 6th July, Time (IST): 11:00 AM
        """
        # Drop internal metadata columns before formatting
        meta_cols = [c for c in df.columns if c.startswith("_")]
        display_df = df.drop(columns=meta_cols, errors="ignore")

        lines: list[str] = []
        cols = list(display_df.columns)

        for _, row in display_df.iterrows():
            parts: list[str] = []
            for col in cols:
                val = row.get(col, "")
                if pd.notna(val) and str(val).strip():
                    parts.append(f"{col}: {str(val).strip()}")
            if parts:
                lines.append(", ".join(parts))

        header = "NareshIT Course Schedule (Latest from Website)"
        timestamp = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return f"{header}\n{timestamp}\n\n" + "\n".join(lines)

    def _save_results(self, df: pd.DataFrame) -> None:
        """Persist scraped data as JSON and text."""
        ensure_dir(settings.scraped_dir)

        # JSON (with source metadata)
        json_data = {
            "source_url": self.url,
            "scraped_at": datetime.now().isoformat(),
            "total_rows": len(df),
            "courses": df.to_dict(orient="records"),
        }
        with open(settings.scraped_json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        logger.info("Saved scraped JSON → %s", settings.scraped_json_path)

        # Text file
        with open(settings.scraped_text_path, "w", encoding="utf-8") as f:
            f.write(self.cleaned_text)
        logger.info("Saved scraped text → %s", settings.scraped_text_path)

    def _load_cached_text(self) -> str:
        """Fall back to the last-saved scraped text file."""
        if settings.scraped_text_path.exists():
            text = settings.scraped_text_path.read_text(encoding="utf-8")
            logger.info("Loaded cached scraped text (%d chars).", len(text))
            return text
        return ""


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def scrape_website() -> str:
    """One-call helper used by the RAG pipeline on startup."""
    scraper = WebsiteScraper()
    return scraper.scrape_and_save()


if __name__ == "__main__":
    result = scrape_website()
    print(f"\n{'='*60}")
    print(f"Scraped text ({len(result)} chars):")
    print(f"{'='*60}")
    print(result[:3000])