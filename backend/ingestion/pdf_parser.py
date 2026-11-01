"""Production-grade PDF parsing engine for RFP intelligence.

Combines:
1. PyMuPDF4LLM structural layout (headings, lists, bolding).
2. Zero-Loss Layout Recovery (recovers text when single-cell border boxes swallow content).
3. PyMuPDF table finder with Banner Splitting (decouples composite government forms).
4. Physical page pinning and scanned page categorization.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import pymupdf
import pymupdf4llm

from app.core.logging import get_logger
from ingestion.exceptions import (
    CorruptedDocumentError,
    EmptyDocumentError,
    PDFParsingError,
    UnsupportedFormatError,
)
from ingestion.models import (
    DiscoveredDocument,
    DocumentType,
    FileType,
    ParsedDocument,
    ParsedPage,
)
from ingestion.ocr import ChandraOCRClient

logger = get_logger(__name__)


class PDFParser:
    """Production-grade PDF parsing engine with layout recovery and table extraction."""

    @staticmethod
    def _extract_headings(md_text: str) -> list[str]:
        """Extracts markdown heading titles (#, ##, ###) from page text."""
        headings: list[str] = []
        for line in md_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                heading_title = stripped.lstrip("#").strip()
                if heading_title:
                    headings.append(heading_title)
        return headings

    @staticmethod
    def _split_composite_tables(
        raw_rows: list[list[str | None]],
    ) -> list[list[list[str]]]:
        """Splits stacked composite form tables at full-width banner rows and drops empty columns.

        Government RFPs often bundle disparate sections (e.g., Contacts, Delivery Info,
        Hardware Specs, Warranty) into a single contiguous bordered grid. This method
        isolates each section into an independent tabular matrix.
        """
        sub_tables: list[list[list[str]]] = []
        current: list[list[str]] = []

        for row in raw_rows:
            cells = [c.strip().replace("\n", " ") if c else "" for c in row]
            non_empty = [c for c in cells if c]
            if not non_empty:
                continue

            # Full-width banner header row (e.g. repeated banner or single title spanning columns)
            is_banner = len(set(non_empty)) == 1 and len(non_empty[0]) > 8
            if is_banner:
                if current:
                    sub_tables.append(current)
                    current = []
                current.append([non_empty[0]])
            else:
                current.append(cells)

        if current:
            sub_tables.append(current)

        # Drop 100% empty columns across each sub-table
        cleaned_tables: list[list[list[str]]] = []
        for tbl in sub_tables:
            if len(tbl) <= 1:
                cleaned_tables.append(tbl)
                continue
            num_cols = max(len(r) for r in tbl)
            cols = [
                [r[i] if i < len(r) else "" for r in tbl[1:]]
                for i in range(num_cols)
            ]
            valid_indices = [i for i, col in enumerate(cols) if any(c for c in col)]
            clean_tbl = [
                [r[i] if i < len(r) else "" for i in valid_indices]
                for r in tbl
            ]
            cleaned_tables.append(clean_tbl)

        return cleaned_tables

    @classmethod
    def _extract_markdown_tables(cls, markdown_text: str) -> list[list[list[str]]]:
        """Extracts structured 2D table grids from Markdown text (e.g., from OCR output)."""
        tables: list[list[str]] = []
        lines = markdown_text.splitlines()
        current_table: list[str] = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                current_table.append(stripped)
            else:
                if len(current_table) >= 3:
                    tables.append(current_table)
                current_table = []
        if len(current_table) >= 3:
            tables.append(current_table)

        parsed_grids: list[list[list[str]]] = []
        for raw_tbl in tables:
            grid: list[list[str]] = []
            for row in raw_tbl:
                cells = [c.strip() for c in row.strip("|").split("|")]
                # Skip markdown separator rows like |---|---|
                if all(re.match(r"^:?-+:?$", c) for c in cells if c):
                    continue
                grid.append(cells)
            if len(grid) >= 2 and max(len(r) for r in grid) >= 2:
                parsed_grids.append(grid)
        return parsed_grids

    @classmethod
    def parse_document(
        cls,
        discovered_doc: DiscoveredDocument,
        max_pages: int | None = None,
        enable_ocr: bool = True,
    ) -> ParsedDocument:
        """Parses a discovered PDF document into a validated ParsedDocument model.

        Args:
            discovered_doc: Validated DiscoveredDocument instance.
            max_pages: Optional maximum number of pages to parse (for fast sampling).
            enable_ocr: Whether to run Chandra 2 OCR on scanned/image pages.

        Returns:
            ParsedDocument with physical page pinning, layout markdown, and tables.

        Raises:
            UnsupportedFormatError: If the document is not a PDF.
            CorruptedDocumentError: If PyMuPDF fails to open or read the file.
            PDFParsingError: If an unhandled parsing error occurs.
        """
        if discovered_doc.file_type != FileType.PDF:
            raise UnsupportedFormatError(
                f"Cannot parse non-PDF file format '{discovered_doc.file_type}' with PDFParser",
                file_path=discovered_doc.file_path,
            )

        return cls.parse_file(
            file_path=discovered_doc.file_path,
            bid_id=discovered_doc.bid_id,
            doc_type=discovered_doc.doc_type,
            addendum_number=discovered_doc.addendum_number,
            max_pages=max_pages,
            enable_ocr=enable_ocr,
        )

    @classmethod
    def parse_file(
        cls,
        file_path: Path | str,
        bid_id: str = "UNKNOWN",
        doc_type: DocumentType = DocumentType.RFP,
        addendum_number: int | None = None,
        max_pages: int | None = None,
        enable_ocr: bool = True,
    ) -> ParsedDocument:
        """Parses a PDF from a filesystem path into a validated ParsedDocument model.

        Args:
            file_path: Filesystem path to the PDF file.
            bid_id: Unique bid identifier.
            doc_type: Procurement document role.
            addendum_number: Addendum number if document is an amendment.
            max_pages: Optional maximum number of pages to parse.
            enable_ocr: Whether to run Chandra 2 OCR on scanned/image pages.

        Returns:
            Validated ParsedDocument with page-level metrics and tables.
        """
        resolved_path = Path(file_path).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {resolved_path}")

        start_time = time.perf_counter()
        logger.info(
            "Starting PDF parse for '%s' (Bid: %s, Role: %s)",
            resolved_path.name,
            bid_id,
            doc_type.value,
        )

        try:
            doc = pymupdf.open(resolved_path)
        except Exception as exc:
            raise CorruptedDocumentError(
                f"Failed to open PDF document: {exc}",
                file_path=resolved_path,
            ) from exc

        with doc:
            doc_len = len(doc)
            if doc_len == 0:
                raise EmptyDocumentError(
                    "PDF file contains 0 pages",
                    file_path=resolved_path,
                )

            pages_to_process = min(doc_len, max_pages) if max_pages else doc_len
            zero_idx_pages = list(range(pages_to_process))

            # 1. Structural layout extraction via pymupdf4llm
            try:
                chunks: list[dict[str, Any]] = pymupdf4llm.to_markdown(
                    doc,
                    pages=zero_idx_pages,
                    page_chunks=True,
                    ignore_code=True,
                    write_images=False,
                )
            except Exception as exc:
                logger.warning(
                    "PyMuPDF4LLM failed on '%s': %s. Falling back to native block extraction.",
                    resolved_path.name,
                    exc,
                )
                chunks = []

            parsed_pages: list[ParsedPage] = []
            total_tables_count = 0
            total_headings_count = 0
            scanned_count = 0
            recovered_count = 0
            ocr_count = 0
            ocr_client = ChandraOCRClient() if enable_ocr else None

            for p_idx in zero_idx_pages:
                page_num = p_idx + 1
                page_obj = doc[p_idx]
                raw_page_text = page_obj.get_text("text").strip()

                chunk_data = chunks[p_idx] if p_idx < len(chunks) else {}
                page_md = chunk_data.get("text", "").strip()

                # Zero-Loss Layout Recovery:
                # If pymupdf4llm drops text due to outer border boxes, recover text
                # from native PyMuPDF blocks to prevent loss of critical specs.
                recovery_applied = False
                if len(page_md) < 50 and len(raw_page_text) >= 50:
                    blocks = page_obj.get_text("blocks")
                    recovered_lines = [b[4].strip() for b in blocks if b[4].strip()]
                    page_md = "\n\n".join(recovered_lines)
                    recovery_applied = True
                    recovered_count += 1
                    logger.debug(
                        "Layout recovery triggered on page %d of '%s' (recovered %d chars)",
                        page_num,
                        resolved_path.name,
                        len(page_md),
                    )

                char_count = len(page_md)
                # Multi-signal scanned/form detection:
                # Identifies scanned images or vector-rendered forms (<40 text chars with drawings or images),
                # preventing wasted OCR calls on truly blank separator pages.
                has_visual_content = bool(page_obj.get_images() or page_obj.get_drawings())
                is_scanned = len(raw_page_text) < 40 and len(page_md) < 40 and has_visual_content
                ocr_applied = False
                if is_scanned:
                    scanned_count += 1
                    # Chandra 2 OCR Integration for scanned/vector-only forms
                    if enable_ocr and ocr_client and ocr_client.is_available:
                        logger.info(
                            "Scanned page %d detected in '%s'; triggering Chandra 2 OCR...",
                            page_num,
                            resolved_path.name,
                        )
                        ocr_md = ocr_client.ocr_page(doc, p_idx)
                        if ocr_md:
                            page_md = ocr_md
                            char_count = len(page_md)
                            ocr_applied = True
                            ocr_count += 1

                headings = cls._extract_headings(page_md)
                total_headings_count += len(headings)

                # 2. Native Table Finder & Banner Splitting
                page_tables: list[list[list[str]]] = []
                try:
                    tabs = page_obj.find_tables()
                    if tabs.tables:
                        for t in tabs.tables:
                            raw_data = t.extract()
                            if raw_data:
                                sub_grids = cls._split_composite_tables(raw_data)
                                for grid in sub_grids:
                                    # Retain structured tables with >= 2 rows and >= 2 columns
                                    if len(grid) >= 2 and max(len(r) for r in grid) >= 2:
                                        page_tables.append(grid)
                except Exception as exc:
                    logger.warning(
                        "Table extraction failed on page %d of '%s': %s",
                        page_num,
                        resolved_path.name,
                        exc,
                    )

                # Fallback to Markdown table extraction when OCR was applied and native vector table finder found none
                if ocr_applied and not page_tables:
                    page_tables = cls._extract_markdown_tables(page_md)

                total_tables_count += len(page_tables)

                # Determine extraction method provenance for auditability
                if ocr_applied:
                    extraction_method = "chandra_ocr"
                elif recovery_applied:
                    extraction_method = "recovered_blocks"
                else:
                    extraction_method = "digital"

                parsed_pages.append(
                    ParsedPage(
                        page_number=page_num,
                        markdown=page_md,
                        char_count=char_count,
                        headings=headings,
                        tables=page_tables,
                        has_tables=bool(page_tables),
                        is_scanned=is_scanned,
                        recovery_applied=recovery_applied,
                        ocr_applied=ocr_applied,
                        extraction_method=extraction_method,
                    )
                )

        duration = round(time.perf_counter() - start_time, 3)
        logger.info(
            "Parsed '%s' in %.3fs (%d pages, %d chars, %d tables, %d headings, %d recovered, %d scanned, %d ocr)",
            resolved_path.name,
            duration,
            len(parsed_pages),
            sum(p.char_count for p in parsed_pages),
            total_tables_count,
            total_headings_count,
            recovered_count,
            scanned_count,
            ocr_count,
        )

        return ParsedDocument(
            bid_id=bid_id,
            file_name=resolved_path.name,
            file_path=resolved_path,
            doc_type=doc_type,
            addendum_number=addendum_number,
            total_pages=len(parsed_pages),
            pages=parsed_pages,
            total_characters=sum(p.char_count for p in parsed_pages),
            total_tables=total_tables_count,
            total_headings=total_headings_count,
            scanned_pages_count=scanned_count,
            recovered_pages_count=recovered_count,
            ocr_pages_count=ocr_count,
            parse_duration_seconds=duration,
        )

