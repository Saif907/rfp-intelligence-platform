"""Agnostic layout complexity router for procurement documents.

Analyzes vector geometry, drawing density, raster image coverage,
interleaving text streams, table topologies, and form input squares
without domain-specific keywords or hardcoded string rules.

Routes pages into three distinct processing tiers:
- 'chandra_balanced': True scans, high-density form checkboxes, or visual templates.
- 'chandra_fast'    : Tables, compound grids, multi-column flows, or code listings.
- 'simple_text'     : Standard linear text streams (processed via local PyMuPDF).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import pymupdf

from app.core.logging import get_logger

logger = get_logger(__name__)


class PageProcessingMode(StrEnum):
    """Target execution engine tier for an individual document page."""

    SIMPLE_TEXT = "simple_text"
    CHANDRA_FAST = "chandra_fast"
    CHANDRA_BALANCED = "chandra_balanced"


class LayoutMetrics(BaseModel):
    """Geometric and visual complexity metrics extracted from a PDF page."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text_length: int = Field(ge=0, description="Total characters of raw extractable text")
    img_coverage: float = Field(ge=0.0, le=1.0, description="Ratio of image bounding box area to page area")
    vector_paths: int = Field(ge=0, description="Count of vector drawing elements")
    tables: int = Field(ge=0, description="Count of detected table topologies")
    interleaving: int = Field(ge=0, description="Spatial Y-overlap count between horizontally distant text blocks")
    small_boxes: int = Field(ge=0, description="Count of square vector paths corresponding to checkboxes/input boxes")


class PageRouteResult(BaseModel):
    """Routing determination and explainability audit trail for a single page."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: int = Field(ge=1, description="1-indexed physical page number")
    mode: PageProcessingMode = Field(..., description="Selected processing tier")
    reason: str = Field(..., description="Audit explanation detailing why the page was routed to this tier")
    metrics: LayoutMetrics = Field(..., description="Extracted geometric metrics")


class DocumentRoutingPlan(BaseModel):
    """Aggregate routing plan for all pages within a PDF document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_name: str = Field(..., description="Source PDF filename")
    total_pages: int = Field(ge=0, description="Total page count")
    page_routes: list[PageRouteResult] = Field(default_factory=list, description="Per-page routing decisions")
    summary: dict[PageProcessingMode, int] = Field(default_factory=dict, description="Count of pages per mode")


class PageLayoutRouter:
    """Production-grade layout router evaluating page geometry and visual density."""

    def __init__(
        self,
        min_scan_img_coverage: float = 0.60,
        max_scan_text_length: int = 200,
        min_form_small_boxes: int = 4,
        min_vector_paths: int = 25,
        min_interleaving_score: int = 3,
    ) -> None:
        self.min_scan_img_coverage = min_scan_img_coverage
        self.max_scan_text_length = max_scan_text_length
        self.min_form_small_boxes = min_form_small_boxes
        self.min_vector_paths = min_vector_paths
        self.min_interleaving_score = min_interleaving_score

    def get_layout_complexity_metrics(self, page: pymupdf.Page) -> LayoutMetrics:
        """Extracts layout geometry, vector density, and visual coverage metrics.

        Computes:
        1. Raster image coverage ratio against total page area.
        2. Raw character count and vector drawing path count.
        3. Spatial interleaving (Y-axis overlap between horizontally separated text blocks).
        4. Geometric table count via native table finder.
        5. Form checkboxes and small square input box elements (6-20 pt squares with aspect ratio ~1.0).
        """
        page_rect = page.rect
        page_area = max(1.0, page_rect.width * page_rect.height)
        text = page.get_text("text").strip()
        drawings = page.get_drawings()
        blocks = [b for b in page.get_text("blocks") if b[4].strip()]

        # 1. Raster Image Coverage Ratio
        image_infos = page.get_image_info(xrefs=True)
        img_area = 0.0
        for img in image_infos:
            b = img.get("bbox")
            if b:
                img_area += max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        img_coverage = min(1.0, img_area / page_area)

        # 2. Text Length and Vector Path Density
        text_length = len(text)
        vector_path_count = len(drawings)

        # 3. Spatial Interleaving (Y-axis overlap among sibling text blocks)
        interleaving_score = 0
        num_blocks = len(blocks)
        for i in range(num_blocks):
            for j in range(i + 1, min(i + 8, num_blocks)):
                b1, b2 = blocks[i], blocks[j]
                y_overlap = max(0.0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                x_distance = abs(b1[0] - b2[0])
                if y_overlap > 20.0 and x_distance > 80.0:
                    interleaving_score += 1

        # 4. Built-in Geometric Table Topology
        try:
            tabs = page.find_tables()
            table_count = len(tabs.tables) if tabs else 0
        except Exception:
            table_count = 0

        # 5. Form Checkboxes / Small Input Squares
        small_box_count = 0
        for d in drawings:
            r = d.get("rect")
            if r:
                w, h = r.width, r.height
                if 6.0 <= w <= 20.0 and 6.0 <= h <= 20.0 and abs(w - h) <= 2.5:
                    small_box_count += 1

        return LayoutMetrics(
            text_length=text_length,
            img_coverage=round(img_coverage, 4),
            vector_paths=vector_path_count,
            tables=table_count,
            interleaving=interleaving_score,
            small_boxes=small_box_count,
        )

    def route_page(self, page: pymupdf.Page, page_number: int = 1) -> PageRouteResult:
        """Routes a single page into an execution tier based on layout geometry."""
        m = self.get_layout_complexity_metrics(page)

        # TIER 2: CHANDRA BALANCED
        # Condition A1: Raster Scanned Page (large image coverage, sparse digital text)
        is_raster_scanned = (
            m.img_coverage > self.min_scan_img_coverage
            and m.text_length < self.max_scan_text_length
        )

        # Condition A2: Vector/Drawing Visual Sheet with zero or sparse digital text (e.g. IRS W-9, forms)
        is_vector_visual = m.text_length < 60 and (
            m.vector_paths > self.min_vector_paths or m.img_coverage > 0.30
        )

        if is_raster_scanned or is_vector_visual:
            reason = (
                f"Visual vector/drawing form sheet (text_len={m.text_length}, paths={m.vector_paths})"
                if is_vector_visual
                else f"Scanned raster sheet (img_cov={m.img_coverage:.2f}, text_len={m.text_length})"
            )
            return PageRouteResult(
                page_number=page_number,
                mode=PageProcessingMode.CHANDRA_BALANCED,
                reason=reason,
                metrics=m,
            )

        # Condition B: Form Checkboxes / Interactive Input Squares
        if m.small_boxes >= self.min_form_small_boxes:
            return PageRouteResult(
                page_number=page_number,
                mode=PageProcessingMode.CHANDRA_BALANCED,
                reason=f"Interactive form/checkbox elements (boxes={m.small_boxes})",
                metrics=m,
            )

        # TIER 1: CHANDRA FAST
        # Condition C: Tables, compound grids, or multi-column/code block layouts
        if m.tables > 0 or m.vector_paths > self.min_vector_paths or m.interleaving >= self.min_interleaving_score:
            return PageRouteResult(
                page_number=page_number,
                mode=PageProcessingMode.CHANDRA_FAST,
                reason=(
                    f"Complex geometry (tables={m.tables}, paths={m.vector_paths}, "
                    f"interleaving={m.interleaving})"
                ),
                metrics=m,
            )

        # TIER 0: SIMPLE TEXT (PyMuPDF local CPU)
        # Condition D: Monotonic single-column text flow with minimal vector ornament
        return PageRouteResult(
            page_number=page_number,
            mode=PageProcessingMode.SIMPLE_TEXT,
            reason=f"Standard linear text (chars={m.text_length}, paths={m.vector_paths})",
            metrics=m,
        )

    def route_document(self, pdf_path: Path | str) -> DocumentRoutingPlan:
        """Analyzes an entire PDF document and generates a complete routing plan."""
        path = Path(pdf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {path}")

        doc = pymupdf.open(path)
        page_routes: list[PageRouteResult] = []
        summary: dict[PageProcessingMode, int] = {
            PageProcessingMode.SIMPLE_TEXT: 0,
            PageProcessingMode.CHANDRA_FAST: 0,
            PageProcessingMode.CHANDRA_BALANCED: 0,
        }

        with doc:
            for idx, page in enumerate(doc):
                res = self.route_page(page, page_number=idx + 1)
                page_routes.append(res)
                summary[res.mode] += 1

        logger.info(
            "Routing plan for '%s' (%d pages): simple_text=%d, chandra_fast=%d, chandra_balanced=%d",
            path.name,
            len(page_routes),
            summary[PageProcessingMode.SIMPLE_TEXT],
            summary[PageProcessingMode.CHANDRA_FAST],
            summary[PageProcessingMode.CHANDRA_BALANCED],
        )

        return DocumentRoutingPlan(
            file_name=path.name,
            total_pages=len(page_routes),
            page_routes=page_routes,
            summary=summary,
        )


# Global default instance
default_router = PageLayoutRouter()

