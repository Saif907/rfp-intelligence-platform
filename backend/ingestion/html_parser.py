"""Production-grade HTML parser for procurement portals and bid overview pages.

Extracts structured procurement metadata, bid information fields, descriptions,
and data tables from HTML portal snapshots (e.g., BidNet Direct) and normalizes
them into validated ParsedDocument models.
"""

from __future__ import annotations

import html
from pathlib import Path
import re
import time
from typing import Any

from app.core.logging import get_logger
from ingestion.exceptions import (
    CorruptedDocumentError,
    EmptyDocumentError,
    HTMLParsingError,
    UnsupportedFormatError,
)
from ingestion.models import (
    DiscoveredDocument,
    DocumentType,
    FileType,
    ParsedDocument,
    ParsedPage,
)

logger = get_logger(__name__)


class HTMLParser:
    """Production parser for bidding portal HTML pages with multi-engine support."""

    @staticmethod
    def _extract_headings(md_text: str) -> list[str]:
        """Extracts markdown heading titles (#, ##, ###) from markdown text."""
        headings: list[str] = []
        for line in md_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                heading_title = stripped.lstrip("#").strip()
                if heading_title:
                    headings.append(heading_title)
        return headings

    @classmethod
    def _clean_text_fragment(cls, raw: str) -> str:
        """Strips HTML tags, collapses whitespace, and unescapes HTML entities."""
        no_tags = re.sub(r"<[^>]+>", " ", raw)
        single_space = re.sub(r"\s+", " ", no_tags).strip()
        return html.unescape(single_space)

    @classmethod
    def _parse_with_bs4(cls, soup: Any) -> tuple[str, dict[str, str], list[list[list[str]]]]:
        """Extracts title, metadata key-values, and tabular grids using BeautifulSoup."""
        # 1. Prune noise tags
        for tag in soup(["script", "style", "svg", "noscript", "nav", "header", "footer"]):
            tag.decompose()

        # 2. Extract Title / H1
        h1_tag = soup.find("h1")
        title_text = cls._clean_text_fragment(h1_tag.get_text()) if h1_tag else ""
        if not title_text and soup.title:
            title_text = cls._clean_text_fragment(soup.title.get_text())

        # 3. Extract BidNet Direct 'mets-field' key-values
        metadata_fields: dict[str, str] = {}
        for field_div in soup.find_all(class_=re.compile(r"mets-field")):
            label_elem = field_div.find(class_=re.compile(r"mets-field-label"))
            body_elem = field_div.find(class_=re.compile(r"mets-field-body"))
            if label_elem and body_elem:
                label = cls._clean_text_fragment(label_elem.get_text())
                body = cls._clean_text_fragment(body_elem.get_text())

                # Clean any trailing JavaScript snippet text from descriptions
                body = re.split(r"var\s+descriptionText_options", body)[0].strip()
                body = re.split(r"See\s+more", body)[0].strip()

                if label and body and label not in metadata_fields:
                    metadata_fields[label] = body

        # 4. Extract standard HTML <table> grids
        extracted_tables: list[list[list[str]]] = []
        for table_tag in soup.find_all("table"):
            grid: list[list[str]] = []
            for tr in table_tag.find_all("tr"):
                row = [cls._clean_text_fragment(c.get_text()) for c in tr.find_all(["th", "td"])]
                if any(row):
                    grid.append(row)
            if len(grid) >= 2 and max(len(r) for r in grid) >= 2:
                extracted_tables.append(grid)

        return title_text, metadata_fields, extracted_tables

    @classmethod
    def _parse_with_native(cls, raw_html: str) -> tuple[str, dict[str, str], list[list[list[str]]]]:
        """Fallback extractor using standard library regex and HTML decoders."""
        # 1. Strip script, style, comments
        cleaned = re.sub(r"<!--.*?-->", "", raw_html, flags=re.DOTALL)
        cleaned = re.sub(r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<noscript\b[^<]*(?:(?!<\/noscript>)<[^<]*)*<\/noscript>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<svg\b[^<]*(?:(?!<\/svg>)<[^<]*)*<\/svg>", "", cleaned, flags=re.IGNORECASE)

        # 2. Extract Title / H1
        h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", cleaned, re.DOTALL | re.IGNORECASE)
        title_text = ""
        if h1_match:
            title_text = cls._clean_text_fragment(h1_match.group(1))
        else:
            title_match = re.search(r"<title[^>]*>(.*?)</title>", cleaned, re.DOTALL | re.IGNORECASE)
            if title_match:
                title_text = cls._clean_text_fragment(title_match.group(1))

        # 3. Extract BidNet Direct 'mets-field' key-values
        field_pattern = (
            r'<span[^>]*class="[^"]*mets-field-label[^"]*"[^>]*>(.*?)</span>\s*'
            r'<div[^>]*class="[^"]*mets-field-body[^"]*"[^>]*>(.*?)</div>'
        )
        matches = re.findall(field_pattern, cleaned, re.DOTALL | re.IGNORECASE)

        metadata_fields: dict[str, str] = {}
        for label_html, body_html in matches:
            label = cls._clean_text_fragment(label_html)
            body = cls._clean_text_fragment(body_html)
            body = re.split(r"var\s+descriptionText_options", body)[0].strip()
            body = re.split(r"See\s+more", body)[0].strip()

            if label and body and label not in metadata_fields:
                metadata_fields[label] = body

        # 4. Extract <table> grids
        table_pattern = r"<table\b[^>]*>(.*?)</table>"
        extracted_tables: list[list[list[str]]] = []
        for tbl_match in re.finditer(table_pattern, cleaned, re.DOTALL | re.IGNORECASE):
            tbl_html = tbl_match.group(1)
            rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", tbl_html, re.DOTALL | re.IGNORECASE)
            grid: list[list[str]] = []
            for r in rows:
                cells = re.findall(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", r, re.DOTALL | re.IGNORECASE)
                row_vals = [cls._clean_text_fragment(c) for c in cells]
                if any(row_vals):
                    grid.append(row_vals)
            if len(grid) >= 2 and max(len(r) for r in grid) >= 2:
                extracted_tables.append(grid)

        return title_text, metadata_fields, extracted_tables

    @classmethod
    def parse_document(cls, discovered_doc: DiscoveredDocument) -> ParsedDocument:
        """Parses a discovered HTML document into a validated ParsedDocument model.

        Args:
            discovered_doc: Validated DiscoveredDocument instance.

        Returns:
            ParsedDocument containing page markdown, key-value table, and metadata.
        """
        if discovered_doc.file_type != FileType.HTML:
            raise UnsupportedFormatError(
                f"Cannot parse non-HTML file format '{discovered_doc.file_type}' with HTMLParser",
                file_path=discovered_doc.file_path,
            )

        return cls.parse_file(
            file_path=discovered_doc.file_path,
            bid_id=discovered_doc.bid_id,
            doc_type=discovered_doc.doc_type,
        )

    @classmethod
    def parse_file(
        cls,
        file_path: Path | str,
        bid_id: str = "UNKNOWN",
        doc_type: DocumentType = DocumentType.BID_PAGE,
    ) -> ParsedDocument:
        """Parses an HTML file from disk into a validated ParsedDocument model.

        Args:
            file_path: Filesystem path to the HTML file.
            bid_id: Unique bid identifier.
            doc_type: Document role (defaults to BID_PAGE).

        Returns:
            Validated ParsedDocument with normalized Markdown and metadata tables.
        """
        resolved_path = Path(file_path).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"HTML file does not exist: {resolved_path}")

        start_time = time.perf_counter()
        logger.info(
            "Starting HTML parse for '%s' (Bid: %s, Role: %s)",
            resolved_path.name,
            bid_id,
            doc_type.value,
        )

        try:
            raw_bytes = resolved_path.read_bytes()
        except Exception as exc:
            raise CorruptedDocumentError(
                f"Failed to read HTML file: {exc}",
                file_path=resolved_path,
            ) from exc

        if len(raw_bytes) == 0:
            raise EmptyDocumentError(
                "HTML file is 0 bytes",
                file_path=resolved_path,
            )

        # Decode with UTF-8 or resilient fallback
        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = raw_bytes.decode("latin-1", errors="ignore")

        # Select parsing engine (BeautifulSoup if available, native fallback otherwise)
        title_text = ""
        metadata_fields: dict[str, str] = {}
        tables: list[list[list[str]]] = []

        try:
            from bs4 import BeautifulSoup  # type: ignore[import-not-found]

            soup = BeautifulSoup(content, "html.parser")
            title_text, metadata_fields, tables = cls._parse_with_bs4(soup)
        except ImportError:
            logger.debug("BeautifulSoup4 not installed; utilizing native HTML parsing engine.")
            title_text, metadata_fields, tables = cls._parse_with_native(content)
        except Exception as exc:
            logger.warning(
                "BeautifulSoup parsing encountered error on '%s': %s. Retrying with native engine.",
                resolved_path.name,
                exc,
            )
            title_text, metadata_fields, tables = cls._parse_with_native(content)

        # Format into clean, high-density Markdown
        md_lines: list[str] = []
        if title_text:
            md_lines.append(f"# {title_text}\n")

        if metadata_fields:
            md_lines.append("## Bid Information Overview\n")
            md_lines.append("| Field | Details |")
            md_lines.append("| :--- | :--- |")
            for key, val in metadata_fields.items():
                if key != "Description":
                    md_lines.append(f"| **{key}** | {val} |")
            md_lines.append("")

            # Keep Description as an explicit section
            if "Description" in metadata_fields:
                md_lines.append("## Bid Description\n")
                md_lines.append(metadata_fields["Description"])
                md_lines.append("")

        page_markdown = "\n".join(md_lines).strip()
        if not page_markdown:
            # If no mets-fields, fall back to plain text content
            plain_text = cls._clean_text_fragment(content)
            page_markdown = f"# {title_text or resolved_path.stem}\n\n{plain_text[:5000]}"

        headings = cls._extract_headings(page_markdown)

        # Convert metadata fields table into a formal 2D table grid for chunkers
        if metadata_fields:
            overview_grid: list[list[str]] = [["Field", "Details"]]
            for k, v in metadata_fields.items():
                if k != "Description":
                    overview_grid.append([k, v])
            tables.insert(0, overview_grid)

        parsed_page = ParsedPage(
            page_number=1,
            markdown=page_markdown,
            char_count=len(page_markdown),
            headings=headings,
            tables=tables,
            has_tables=bool(tables),
            is_scanned=False,
            recovery_applied=False,
        )

        duration = round(time.perf_counter() - start_time, 3)
        logger.info(
            "Parsed HTML '%s' in %.3fs (%d chars, %d fields, %d tables)",
            resolved_path.name,
            duration,
            len(page_markdown),
            len(metadata_fields),
            len(tables),
        )

        return ParsedDocument(
            bid_id=bid_id,
            file_name=resolved_path.name,
            file_path=resolved_path,
            doc_type=doc_type,
            addendum_number=None,
            total_pages=1,
            pages=[parsed_page],
            total_characters=len(page_markdown),
            total_tables=len(tables),
            total_headings=len(headings),
            scanned_pages_count=0,
            recovered_pages_count=0,
            parse_duration_seconds=duration,
        )

