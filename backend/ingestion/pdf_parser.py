"""Production-grade PDF parsing engine for procurement intelligence.

Combines:
1. Agnostic Page Layout Router for intelligent tier selection (simple_text, chandra_fast, chandra_balanced).
2. Chandra 2 Vision OCR with HTML extraction (preserves tables without merged cell duplication).
3. Geometric y0-coordinate block and table interleaving (keeps tables anchored to their section headers).
4. Automated layout cleaning (strips repeated running headers, repairs isolated bullets, deduplicates rows).
5. Physical page pinning and complete audit provenance.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import pymupdf
import pymupdf4llm

from app.core.logging import get_logger
from ingestion.cleaner import DocumentCleaner
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
from ingestion.ocr import ChandraOCRClient, HTMLLayoutExtractor
from ingestion.router import (
    PageLayoutRouter,
    PageProcessingMode,
    default_router,
)

logger = get_logger(__name__)


class PDFParser:
    """Production PDF parsing engine combining PyMuPDF layout, vertical interleaving, and Chandra OCR."""

    @staticmethod
    def _extract_headings(md_text: str) -> list[str]:
        """Extracts markdown heading titles (#, ##, ###) from page text."""
        headings: list[str] = []
        for line in md_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                heading_title = stripped.lstrip("#").strip()
                if heading_title and len(heading_title) < 150:
                    headings.append(heading_title)
        return headings

    @staticmethod
    def _is_decorative_table(
        bbox: tuple[float, float, float, float],
        row_count: int,
        col_count: int,
        page_rect: pymupdf.Rect,
    ) -> bool:
        """Determines if a detected table is actually an aesthetic page border/frame container."""
        tbl_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        page_area = max(1.0, page_rect.width * page_rect.height)
        h = bbox[3] - bbox[1]
        avg_row_h = h / max(1, row_count)
        cov = tbl_area / page_area
        return cov > 0.45 and col_count <= 3 and avg_row_h > 65.0

    @staticmethod
    def _interleave_blocks_and_tables(
        page_obj: pymupdf.Page,
        raw_tables: list[list[list[str]]],
        table_bboxes: list[tuple[float, float, float, float]],
    ) -> str:
        """Interleaves text blocks and native tables by vertical y0 coordinate.

        Guarantees that tables remain anchored immediately beneath their respective
        section or line item titles instead of being dumped at the bottom of the page.
        """
        blocks = [b for b in page_obj.get_text("blocks") if b[4].strip()]
        items: list[tuple[float, str, Any]] = []

        # Add text blocks (filtering out running headers/footers)
        for b in blocks:
            text = b[4].strip()
            if any(pat.search(text) for pat in DocumentCleaner.RUNNING_HEADER_PATTERNS):
                continue
            items.append((b[1], "text", text))

        # Add tables
        for bbox, tbl in zip(table_bboxes, raw_tables):
            items.append((bbox[1], "table", tbl))

        items.sort(key=lambda x: x[0])

        elements: list[str] = []
        for _, itype, data in items:
            if itype == "text":
                elements.append(data)
            else:
                md_tbl = HTMLLayoutExtractor._grid_to_markdown(data)
                if md_tbl:
                    elements.append(md_tbl)

        return "\n\n".join(elements)

    @classmethod
    def parse_document(
        cls,
        discovered_doc: DiscoveredDocument,
        max_pages: int | None = None,
        enable_ocr: bool = True,
        router: PageLayoutRouter | None = None,
    ) -> ParsedDocument:
        """Parses a discovered PDF document into a validated ParsedDocument model."""
        if discovered_doc.file_type != FileType.PDF:
            raise UnsupportedFormatError(
                f"Cannot parse non-PDF format '{discovered_doc.file_type}' with PDFParser",
                file_path=discovered_doc.file_path,
            )

        return cls.parse_file(
            file_path=discovered_doc.file_path,
            bid_id=discovered_doc.bid_id,
            doc_type=discovered_doc.doc_type,
            addendum_number=discovered_doc.addendum_number,
            max_pages=max_pages,
            enable_ocr=enable_ocr,
            router=router,
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
        router: PageLayoutRouter | None = None,
    ) -> ParsedDocument:
        """Parses a PDF into a validated ParsedDocument model with layout-anchored tables."""
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

        page_router = router or default_router
        ocr_client = ChandraOCRClient() if enable_ocr else None

        with doc:
            doc_len = len(doc)
            if doc_len == 0:
                raise EmptyDocumentError(
                    "PDF file contains 0 pages",
                    file_path=resolved_path,
                )

            pages_to_process = min(doc_len, max_pages) if max_pages else doc_len
            zero_idx_pages = list(range(pages_to_process))

            # Baseline structural layout extraction via pymupdf4llm
            try:
                llm_chunks: list[dict[str, Any]] = pymupdf4llm.to_markdown(
                    doc,
                    pages=zero_idx_pages,
                    page_chunks=True,
                    ignore_code=True,
                    write_images=False,
                )
            except Exception as exc:
                logger.warning("PyMuPDF4LLM failed on '%s': %s", resolved_path.name, exc)
                llm_chunks = []

            parsed_pages: list[ParsedPage] = []
            total_tables_count = 0
            total_headings_count = 0
            scanned_count = 0
            recovered_count = 0
            ocr_count = 0

            for p_idx in zero_idx_pages:
                page_num = p_idx + 1
                page_obj = doc[p_idx]
                raw_page_text = page_obj.get_text("text").strip()

                chunk_data = llm_chunks[p_idx] if p_idx < len(llm_chunks) else {}
                llm_md = chunk_data.get("text", "").strip()

                # Evaluate layout geometry via agnostic router
                route_result = page_router.route_page(page_obj, page_number=page_num)
                target_mode = route_result.mode
                metrics = route_result.metrics

                page_md = ""
                page_tables: list[list[list[str]]] = []
                extraction_method = "digital"
                recovery_applied = False
                ocr_applied = False
                has_zero_text = metrics.text_length < 60
                has_visual_content = metrics.vector_paths > 20 or metrics.img_coverage > 0.30
                is_scanned = (metrics.img_coverage > 0.60 and metrics.text_length < 200) or (
                    has_zero_text and has_visual_content
                )

                # 1. Check if page warrants Chandra OCR (scanned, complex forms, or border-box text drop)
                border_box_swallowed = len(llm_md) < 50 and len(raw_page_text) >= 50
                warrants_ocr = (
                    is_scanned
                    or (target_mode == PageProcessingMode.CHANDRA_BALANCED)
                    or (target_mode == PageProcessingMode.CHANDRA_FAST and metrics.text_length < 100)
                    or border_box_swallowed
                )

                if enable_ocr and ocr_client and ocr_client.is_available and warrants_ocr:
                    ocr_mode = "balanced" if (target_mode == PageProcessingMode.CHANDRA_BALANCED or is_scanned) else "fast"
                    logger.debug(
                        "Routing page %d to Chandra OCR (mode=%s, reason=%s)",
                        page_num,
                        ocr_mode,
                        route_result.reason,
                    )
                    ocr_result = ocr_client.ocr_page(
                        doc,
                        page_index_0based=p_idx,
                        mode=ocr_mode,
                        exclude_headers_footers=True,
                    )
                    if ocr_result and len(ocr_result.strip()) > 20:
                        page_md = ocr_result.strip()
                        ocr_applied = True
                        ocr_count += 1
                        extraction_method = f"chandra_{ocr_mode}"

                # 2. Native Table Detection & Interleaving Fallback
                try:
                    tabs = page_obj.find_tables()
                    native_tables: list[list[list[str]]] = []
                    table_bboxes: list[tuple[float, float, float, float]] = []
                    if tabs and tabs.tables:
                        for t in tabs.tables:
                            if cls._is_decorative_table(t.bbox, t.row_count, t.col_count, page_obj.rect):
                                continue
                            data = t.extract()
                            if data and len(data) >= 1:
                                # Clean cells
                                clean_grid = [
                                    [c.strip().replace("\n", " ") if c else "" for c in row]
                                    for row in data
                                ]
                                if any(any(c for c in r) for r in clean_grid):
                                    native_tables.append(clean_grid)
                                    table_bboxes.append(t.bbox)
                except Exception:
                    native_tables = []
                    table_bboxes = []

                # If OCR was not applied, assemble from digital/PyMuPDF blocks
                if not page_md:
                    if border_box_swallowed:
                        # Zero-loss layout recovery with vertical interleaving
                        page_md = cls._interleave_blocks_and_tables(
                            page_obj, native_tables, table_bboxes
                        )
                        recovery_applied = True
                        recovered_count += 1
                        extraction_method = "recovered_blocks"
                    elif native_tables and len(native_tables) > 1:
                        # Interleave tables vertically so pricing line tables don't get displaced
                        page_md = cls._interleave_blocks_and_tables(
                            page_obj, native_tables, table_bboxes
                        )
                        extraction_method = "digital"
                    else:
                        page_md = llm_md if llm_md else raw_page_text
                        extraction_method = "digital"

                    page_tables = native_tables

                if is_scanned:
                    scanned_count += 1

                # 3. Post-Extraction Sanitization & Cleaning
                page_md = DocumentCleaner.clean_page_markdown(page_md)

                # Extract headings
                headings = cls._extract_headings(page_md)
                total_headings_count += len(headings)
                total_tables_count += len(page_tables)

                parsed_pages.append(
                    ParsedPage(
                        page_number=page_num,
                        markdown=page_md,
                        char_count=len(page_md),
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
        total_chars = sum(p.char_count for p in parsed_pages)

        logger.info(
            "Parsed '%s' in %.3fs (%d pages, %d chars, %d tables, %d headings, %d ocr, %d recovered)",
            resolved_path.name,
            duration,
            len(parsed_pages),
            total_chars,
            total_tables_count,
            total_headings_count,
            ocr_count,
            recovered_count,
        )

        return ParsedDocument(
            bid_id=bid_id,
            file_name=resolved_path.name,
            file_path=resolved_path,
            doc_type=doc_type,
            addendum_number=addendum_number,
            total_pages=len(parsed_pages),
            pages=parsed_pages,
            total_characters=total_chars,
            total_tables=total_tables_count,
            total_headings=total_headings_count,
            scanned_pages_count=scanned_count,
            recovered_pages_count=recovered_count,
            ocr_pages_count=ocr_count,
            parse_duration_seconds=duration,
        )
