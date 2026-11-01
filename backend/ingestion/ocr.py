"""Chandra 2 OCR client integration for layout-aware document extraction.

Interacts with Datalab's Chandra 2 Document Intelligence API to convert scanned,
handwritten, or vector-only PDF forms into structured Markdown with preserved tables.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import httpx
import pymupdf

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ChandraOCRClient:
    """Production client for Datalab Chandra 2 Document Intelligence API with SHA-256 disk cache."""

    BASE_URL = "https://www.datalab.to/api/v1/convert"

    def __init__(self, api_key: str | None = None, cache_dir: Path | str | None = None) -> None:
        self.api_key = api_key or settings.OCR_API_KEY
        if not self.api_key:
            logger.warning("No OCR_API_KEY found in settings or environment.")

        # Initialize disk cache under backend/.cache/ocr
        if cache_dir is None:
            self.cache_dir = Path(__file__).resolve().parent.parent / ".cache" / "ocr"
        else:
            self.cache_dir = Path(cache_dir).resolve()

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_available(self) -> bool:
        """True if an OCR API key is configured or persistent cache is present."""
        return bool(self.api_key) or self.cache_dir.exists()

    @classmethod
    def compute_page_fingerprint(cls, doc: pymupdf.Document, page_index_0based: int) -> str:
        """Computes a deterministic, content-based hash for a PDF page."""
        page = doc[page_index_0based]
        try:
            page_stream = page.read_contents()
        except Exception:
            page_stream = b""

        if not page_stream:
            drawings_count = len(page.get_drawings())
            images_count = len(page.get_images())
            page_stream = f"{page.rect}:{drawings_count}:{images_count}".encode("utf-8")

        stream_hash = hashlib.sha256(page_stream).hexdigest()
        doc_identifier = Path(doc.name).stem if doc.name else "stream"
        key_str = f"{doc_identifier}:page_{page_index_0based + 1}:{stream_hash}"
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def ocr_page(
        self,
        doc: pymupdf.Document,
        page_index_0based: int,
        timeout_seconds: float = 60.0,
        mode: str = "accurate",
    ) -> str:
        """Sends a single scanned PDF page to Chandra 2 and returns layout Markdown.

        Uses content hashing to avoid redundant API billable requests on repeated runs.

        Args:
            doc: Open PyMuPDF Document instance.
            page_index_0based: 0-indexed page number to extract and process.
            timeout_seconds: Maximum total wait time for OCR polling.
            mode: Inference mode ('fast', 'balanced', or 'accurate').

        Returns:
            Layout-aware Markdown text with preserved tables and headers.
        """
        page_num_1based = page_index_0based + 1

        # Check deterministic SHA-256 disk cache first
        content_hash = self.compute_page_fingerprint(doc, page_index_0based)
        cache_file = self.cache_dir / f"{content_hash}.md"
        if cache_file.exists():
            cached_text = cache_file.read_text(encoding="utf-8").strip()
            if cached_text:
                logger.info(
                    "Retrieved OCR layout for page %d from deterministic disk cache (%d chars)",
                    page_num_1based,
                    len(cached_text),
                )
                return cached_text

        # 1. Slice target page into an isolated in-memory PDF
        single_doc = pymupdf.open()
        single_doc.insert_pdf(doc, from_page=page_index_0based, to_page=page_index_0based)
        pdf_bytes = single_doc.tobytes()
        single_doc.close()

        if not self.is_available:
            logger.warning("Chandra OCR called but no OCR_API_KEY is available and cache missed.")
            return ""

        logger.info(
            "Submitting page %d (%d bytes) to Chandra 2 OCR (mode: %s)...",
            page_num_1based,
            len(pdf_bytes),
            mode,
        )

        headers = {"X-Api-Key": self.api_key}
        data = {"output_format": "markdown", "mode": mode}
        files = {"file": (f"page_{page_num_1based}.pdf", pdf_bytes, "application/pdf")}

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(self.BASE_URL, headers=headers, data=data, files=files)
                if resp.status_code != 200:
                    logger.error(
                        "Chandra OCR request failed (HTTP %d): %s",
                        resp.status_code,
                        resp.text[:300],
                    )
                    return ""

                res_json = resp.json()
                request_id = res_json.get("request_id")
                check_url = res_json.get("request_check_url") or f"{self.BASE_URL}/{request_id}"

                # 2. Poll for completion with exponential backoff
                start_time = time.perf_counter()
                delay = 2.0
                while (time.perf_counter() - start_time) < timeout_seconds:
                    time.sleep(delay)
                    check_resp = client.get(check_url, headers=headers)
                    if check_resp.status_code != 200:
                        logger.warning("Polling error (HTTP %d)", check_resp.status_code)
                        continue

                    check_data = check_resp.json()
                    status = check_data.get("status")
                    logger.debug("Chandra OCR poll for page %d: %s", page_num_1based, status)

                    if status == "complete":
                        md_result = check_data.get("markdown", "").strip()
                        duration = round(time.perf_counter() - start_time, 2)
                        logger.info(
                            "Chandra OCR succeeded for page %d in %.2fs (%d chars)",
                            page_num_1based,
                            duration,
                            len(md_result),
                        )
                        # Persist to disk cache
                        cache_file.write_text(md_result, encoding="utf-8")
                        return md_result
                    elif status == "failed":
                        logger.error("Chandra OCR failed for page %d: %s", page_num_1based, check_data)
                        return ""

                    delay = min(delay * 1.3, 5.0)

                logger.warning("Chandra OCR timed out after %.1fs on page %d", timeout_seconds, page_num_1based)
                return ""

        except Exception as exc:
            logger.error("Exception during Chandra OCR on page %d: %s", page_num_1based, exc)
            return ""

