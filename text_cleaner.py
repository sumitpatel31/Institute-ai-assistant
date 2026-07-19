"""
text_cleaner.py — Text cleaning utilities for the RAG pipeline.

Removes HTML artifacts, extra whitespace, duplicate lines, null values,
and other noise from both scraped website data and extracted PDF text.
"""

from __future__ import annotations

import re
from typing import Sequence


class TextCleaner:
    """Stateless text-cleaning helper."""

    # Patterns to strip out entirely
    _RE_HTML_TAG = re.compile(r"<[^>]+>")
    _RE_MULTIPLE_SPACES = re.compile(r" {2,}")
    _RE_BLANK_LINES = re.compile(r"\n{3,}")

    # Common footer / boilerplate lines to drop (case-insensitive)
    _BOILERPLATE_PATTERNS: Sequence[str] = (
        "copyright",
        "all rights reserved",
        "powered by",
        "privacy policy",
        "terms of service",
        "subscribe to our",
        "follow us on",
        "share this",
    )

    def clean_text(self, text: str) -> str:
        """
        Full cleaning pipeline applied in order:
        1. Strip HTML tags
        2. Normalise whitespace
        3. Remove boilerplate lines
        4. Deduplicate consecutive lines
        5. Trim
        """
        if not text:
            return ""

        text = self._strip_html_tags(text)
        text = self._normalise_whitespace(text)
        text = self._remove_boilerplate(text)
        text = self._deduplicate_lines(text)
        text = text.strip()
        return text

    def clean_table_cell(self, value: str) -> str:
        """Clean a single table cell value."""
        if not value:
            return ""
        value = self._strip_html_tags(value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _strip_html_tags(self, text: str) -> str:
        return self._RE_HTML_TAG.sub("", text)

    def _normalise_whitespace(self, text: str) -> str:
        # Replace various whitespace chars with normal space
        text = re.sub(r"[\t\r\f\v]", " ", text)
        # Collapse multiple spaces into one
        text = self._RE_MULTIPLE_SPACES.sub(" ", text)
        # Collapse 3+ newlines into 2
        text = self._RE_BLANK_LINES.sub("\n\n", text)
        return text

    def _remove_boilerplate(self, text: str) -> str:
        lines = text.split("\n")
        cleaned: list[str] = []
        for line in lines:
            lower = line.strip().lower()
            if any(bp in lower for bp in self._BOILERPLATE_PATTERNS):
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    def _deduplicate_lines(self, text: str) -> str:
        """Remove consecutive duplicate lines (keep first occurrence)."""
        lines = text.split("\n")
        seen: str | None = None
        deduped: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped == seen:
                continue
            if stripped:
                seen = stripped
            deduped.append(line)
        return "\n".join(deduped)