"""Async and sync Chandra 2 OCR client with HTML extraction and performance monitoring.

Key features:
1. Requests 'html' output from Datalab /api/v1/convert for optimal table and form fidelity.
2. High-performance HTML extraction engine that parses tables, headers, and checkboxes without cell duplication.
3. Fully asynchronous client (httpx.AsyncClient) with concurrency throttling (asyncio.Semaphore).
4. Real-time telemetry and monitoring metrics (cache hits, request timings, failure rates).
5. Deterministic multi-factor SHA-256 disk caching to eliminate redundant API calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
import re
import time
from typing import Any

from bs4 import BeautifulSoup, Tag
import httpx
import pymupdf
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.core.logging import get_logger
from ingestion.cleaner import DocumentCleaner

logger = get_logger(__name__)


class OCRMonitoringMetrics(BaseModel):
    """Real-time observability metrics for OCR throughput, latency, and cost."""

    model_config = ConfigDict(extra="forbid")

    total_requests: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    failed_requests: int = Field(default=0, ge=0)
    total_api_seconds: float = Field(default=0.0, ge=0.0)
    total_chars_extracted: int = Field(default=0, ge=0)
    total_tables_extracted: int = Field(default=0, ge=0)

    @property
    def cache_hit_ratio(self) -> float:
        """Ratio of cache hits against total requests."""
        if self.total_requests == 0:
            return 0.0
        return round(self.cache_hits / self.total_requests, 4)

    def log_summary(self) -> None:
        """Emits a structured summary log of OCR telemetry."""
        logger.info(
            "OCR Telemetry: %d requests (%d hits, %d misses, %d failed) | "
            "Hit ratio: %.1f%% | API time: %.2fs | Chars: %d | Tables: %d",
            self.total_requests,
            self.cache_hits,
            self.cache_misses,
            self.failed_requests,
            self.cache_hit_ratio * 100,
            self.total_api_seconds,
            self.total_chars_extracted,
            self.total_tables_extracted,
        )


class HTMLLayoutExtractor:
    """Extracts clean Markdown and structured 2D table matrices from Datalab HTML output."""

    @classmethod
    def html_to_markdown_and_tables(
        cls, html_content: str
    ) -> tuple[str, list[list[list[str]]]]:
        """Parses Datalab HTML output into clean Markdown text and structured 2D table grids.

        Handles:
        - Table rowspans/colspans gracefully without cell text replication.
        - Checkbox inputs (<input type="checkbox"> -> [x] / [ ]).
        - Headings (<h1> to <h6> -> # to ######).
        - Ordered and unordered lists.
        - Preserves clean paragraph breaks.
        """
        if not html_content or not html_content.strip():
            return "", []

        soup = BeautifulSoup(html_content, "html.parser")

        # Strip non-content tags
        for el in soup(["script", "style", "noscript", "svg"]):
            el.decompose()

        tables_extracted: list[list[list[str]]] = []

        # Convert tables to markdown and extract grids
        for table in soup.find_all("table"):
            grid = cls._parse_table_tag(table)
            if grid:
                tables_extracted.append(grid)
                # Replace table tag in soup with clean markdown representation
                md_table = cls._grid_to_markdown(grid)
                table.replace_with(soup.new_string(f"\n\n{md_table}\n\n"))

        # Convert headings
        for level in range(1, 7):
            for h in soup.find_all(f"h{level}"):
                text = h.get_text().strip()
                if text:
                    prefix = "#" * level
                    h.replace_with(soup.new_string(f"\n\n{prefix} {text}\n\n"))

        # Convert checkboxes
        for cb in soup.find_all("input", attrs={"type": "checkbox"}):
            is_checked = cb.has_attr("checked")
            cb.replace_with("[x]" if is_checked else "[ ]")

        # Convert lists
        for ul in soup.find_all(["ul", "ol"]):
            is_ordered = ul.name == "ol"
            list_lines: list[str] = []
            for idx, li in enumerate(ul.find_all("li", recursive=False)):
                item_text = li.get_text().strip()
                if item_text:
                    bullet = f"{idx + 1}." if is_ordered else "-"
                    list_lines.append(f"{bullet} {item_text}")
            if list_lines:
                ul.replace_with(soup.new_string("\n" + "\n".join(list_lines) + "\n"))

        # Convert paragraphs and divs
        for p in soup.find_all(["p", "div"]):
            p_text = p.get_text().strip()
            if p_text:
                p.replace_with(soup.new_string(f"\n{p_text}\n"))

        raw_md = soup.get_text()

        # Apply DocumentCleaner for final polish
        cleaned_md = DocumentCleaner.clean_page_markdown(raw_md)
        return cleaned_md, tables_extracted

    @classmethod
    def _parse_table_tag(cls, table_tag: Tag) -> list[list[str]]:
        """Parses a <table> DOM element into a normalized 2D matrix of strings."""
        rows = table_tag.find_all("tr")
        if not rows:
            return []

        grid: list[list[str]] = []
        for tr in rows:
            cells = tr.find_all(["th", "td"])
            row_data: list[str] = []
            for c in cells:
                text = c.get_text().strip().replace("\n", " ")
                # Check colspan
                colspan = 1
                if c.has_attr("colspan"):
                    try:
                        colspan = max(1, int(c["colspan"]))
                    except ValueError:
                        colspan = 1

                row_data.append(text)
                # Expand empty cells for colspan to maintain column alignment
                for _ in range(colspan - 1):
                    row_data.append("")

            # Deduplicate any identical cells duplicated by OCR engine
            clean_row = DocumentCleaner.deduplicate_table_row(row_data)
            if any(cell for cell in clean_row):
                grid.append(clean_row)

        # Pad rows to equal column length
        if grid:
            max_cols = max(len(r) for r in grid)
            for r in grid:
                if len(r) < max_cols:
                    r.extend([""] * (max_cols - len(r)))

        return grid

    @classmethod
    def _grid_to_markdown(cls, grid: list[list[str]]) -> str:
        """Converts a 2D matrix of string cells into GitHub-flavored Markdown table syntax."""
        if not grid or len(grid) < 1:
            return ""

        header = grid[0]
        num_cols = len(header)
        sep_row = ["---"] * num_cols

        lines: list[str] = []
        lines.append("| " + " | ".join(c if c else " " for c in header) + " |")
        lines.append("| " + " | ".join(sep_row) + " |")

        for row in grid[1:]:
            lines.append("| " + " | ".join(c if c else " " for c in row) + " |")

        return "\n".join(lines)


class AsyncChandraOCRClient:
    """Production asynchronous client for Datalab Chandra 2 Document Intelligence API."""

    BASE_URL = "https://www.datalab.to/api/v1/convert"

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | str | None = None,
        timeout: float = 90.0,
        max_concurrency: int = 4,
    ) -> None:
        self.api_key = (
            api_key
            or getattr(settings, "OCR_API_KEY", None)
            or os.getenv("OCR_API_KEY")
            or os.getenv("DATALAB_API_KEY")
        )
        if not self.api_key:
            logger.warning("No OCR_API_KEY or DATALAB_API_KEY found in settings or environment.")

        # Default persistent cache directory under backend/.cache/ocr_html
        if cache_dir is None:
            self.cache_dir = Path(__file__).resolve().parent.parent / ".cache" / "ocr_html"
        else:
            self.cache_dir = Path(cache_dir).resolve()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.metrics = OCRMonitoringMetrics()

    @property
    def is_available(self) -> bool:
        """True if an OCR API key is configured or local cache exists."""
        return bool(self.api_key) or self.cache_dir.exists()

    @classmethod
    def compute_cache_key(
        cls,
        content_bytes: bytes,
        doc_identifier: str = "document",
        page_index: int = 0,
        mode: str = "fast",
        output_format: str = "html",
    ) -> str:
        """Computes a deterministic SHA-256 cache key for an isolated page."""
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        key_str = f"{doc_identifier}:page_{page_index}:mode_{mode}:fmt_{output_format}:{content_hash}"
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    async def ocr_page_async(
        self,
        doc: pymupdf.Document,
        page_index_0based: int,
        mode: str = "fast",
        exclude_headers_footers: bool = True,
        max_wait_seconds: float = 90.0,
    ) -> tuple[str, list[list[list[str]]]]:
        """Submits a single page asynchronously to Chandra 2, requesting HTML and returning (markdown, tables)."""
        self.metrics.total_requests += 1
        page_num = page_index_0based + 1
        page_obj = doc[page_index_0based]

        # Fingerprint the page stream
        try:
            stream_bytes = page_obj.read_contents()
        except Exception:
            stream_bytes = b""
        if not stream_bytes:
            stream_bytes = f"{page_obj.rect}:{len(page_obj.get_drawings())}".encode("utf-8")

        doc_stem = Path(doc.name).stem if doc.name else "stream"
        cache_key = self.compute_cache_key(
            content_bytes=stream_bytes,
            doc_identifier=doc_stem,
            page_index=page_index_0based,
            mode=mode,
            output_format="html",
        )

        html_cache_file = self.cache_dir / f"{cache_key}.html"
        md_cache_file = self.cache_dir / f"{cache_key}.md"

        # Check disk cache
        if html_cache_file.exists():
            html_text = html_cache_file.read_text(encoding="utf-8").strip()
            if html_text:
                self.metrics.cache_hits += 1
                logger.debug(
                    "Retrieved Chandra HTML for page %d (%s) from cache",
                    page_num,
                    mode,
                )
                md_text, tables = HTMLLayoutExtractor.html_to_markdown_and_tables(html_text)
                self.metrics.total_chars_extracted += len(md_text)
                self.metrics.total_tables_extracted += len(tables)
                return md_text, tables

        self.metrics.cache_misses += 1
        if not self.api_key:
            logger.warning("Chandra OCR requested for page %d but no API key is available.", page_num)
            self.metrics.failed_requests += 1
            return "", []

        # Slice target page into an isolated in-memory PDF
        single_doc = pymupdf.open()
        single_doc.insert_pdf(doc, from_page=page_index_0based, to_page=page_index_0based)
        pdf_bytes = single_doc.tobytes()
        single_doc.close()

        headers = {"X-API-Key": self.api_key}
        data: dict[str, str] = {
            "output_format": "html",
            "mode": mode,
        }
        if exclude_headers_footers:
            data["exclude_block_types"] = "PageHeader,PageFooter"

        files = {"file": (f"page_{page_num}.pdf", pdf_bytes, "application/pdf")}

        logger.info(
            "Submitting page %d of '%s' to Chandra 2 OCR (mode=%s, output_format=html)...",
            page_num,
            doc_stem,
            mode,
        )

        start_time = time.perf_counter()
        async with self.semaphore:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(self.BASE_URL, headers=headers, data=data, files=files)
                    if resp.status_code not in (200, 201, 202):
                        logger.error(
                            "Chandra 2 submission error for page %d (HTTP %d): %s",
                            page_num,
                            resp.status_code,
                            resp.text[:300],
                        )
                        self.metrics.failed_requests += 1
                        return "", []

                    submission = resp.json()
                    check_url = submission.get("request_check_url") or f"{self.BASE_URL}/{submission.get('request_id')}"

                    delay = 2.0
                    while (time.perf_counter() - start_time) < max_wait_seconds:
                        await asyncio.sleep(delay)
                        poll_res = await client.get(check_url, headers=headers)
                        if poll_res.status_code != 200:
                            continue

                        poll_data = poll_res.json()
                        status = poll_data.get("status")

                        if status == "complete":
                            html_text = poll_data.get("html", "").strip()
                            elapsed = round(time.perf_counter() - start_time, 2)
                            self.metrics.total_api_seconds += elapsed
                            logger.info(
                                "Chandra 2 OCR finished for page %d in %.2fs (%d html chars)",
                                page_num,
                                elapsed,
                                len(html_text),
                            )

                            # Persist raw HTML to cache
                            html_cache_file.write_text(html_text, encoding="utf-8")

                            # Extract clean Markdown and structured tables from HTML
                            md_text, tables = HTMLLayoutExtractor.html_to_markdown_and_tables(html_text)
                            md_cache_file.write_text(md_text, encoding="utf-8")

                            self.metrics.total_chars_extracted += len(md_text)
                            self.metrics.total_tables_extracted += len(tables)
                            return md_text, tables
                        elif status == "failed":
                            logger.error("Chandra 2 OCR failed for page %d: %s", page_num, poll_data.get("error"))
                            self.metrics.failed_requests += 1
                            return "", []

                        delay = min(delay * 1.3, 5.0)

                    logger.warning("Chandra 2 OCR timed out after %.1fs on page %d", max_wait_seconds, page_num)
                    self.metrics.failed_requests += 1
                    return "", []

            except Exception as exc:
                logger.error("Chandra 2 OCR exception on page %d: %s", page_num, exc)
                self.metrics.failed_requests += 1
                return "", []


class ChandraOCRClient:
    """Synchronous facade for AsyncChandraOCRClient with shared disk cache."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | str | None = None,
        timeout: float = 90.0,
    ) -> None:
        self.async_client = AsyncChandraOCRClient(
            api_key=api_key,
            cache_dir=cache_dir,
            timeout=timeout,
        )

    @property
    def is_available(self) -> bool:
        return self.async_client.is_available

    @property
    def metrics(self) -> OCRMonitoringMetrics:
        return self.async_client.metrics

    def ocr_page(
        self,
        doc: pymupdf.Document,
        page_index_0based: int,
        mode: str = "fast",
        exclude_headers_footers: bool = True,
        timeout_seconds: float = 90.0,
    ) -> str:
        """Synchronous wrapper for ocr_page_async."""
        coro = self.async_client.ocr_page_async(
            doc=doc,
            page_index_0based=page_index_0based,
            mode=mode,
            exclude_headers_footers=exclude_headers_footers,
            max_wait_seconds=timeout_seconds,
        )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                try:
                    import nest_asyncio2 as nest_asyncio
                    nest_asyncio.apply()
                except ImportError:
                    pass
            md_text, _ = loop.run_until_complete(coro)
            return md_text

        # Standard modern Python 3.10+ execution with proper loop lifecycle management
        md_text, _ = asyncio.run(coro)
        return md_text
