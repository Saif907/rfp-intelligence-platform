"""Unit tests for the layout router, Chandra 2 OCR client, HTML layout extractor, and cleaner."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pymupdf

from ingestion.cleaner import DocumentCleaner
from ingestion.ocr import (
    AsyncChandraOCRClient,
    ChandraOCRClient,
    HTMLLayoutExtractor,
    OCRMonitoringMetrics,
)
from ingestion.router import (
    DocumentRoutingPlan,
    LayoutMetrics,
    PageLayoutRouter,
    PageProcessingMode,
    PageRouteResult,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
BID1_PDF = DATA_DIR / "Bid1" / "JA-207652 Student and Staff Computing Devices FINAL.pdf"
BID2_PDF = DATA_DIR / "Bid2" / "PORFP_-_Dell_Laptop_Final.pdf"


class TestDocumentCleaner(unittest.TestCase):
    """Test suite for DocumentCleaner text sanitization and table deduplication."""

    def test_strip_running_headers_and_footers(self) -> None:
        raw_text = (
            "Request For Proposal 168884 JA-207652 Student and Staff Computing Devices\n"
            "Dallas ISD rev 2.0\n"
            "Page 4 of 40\n"
            "Real content line."
        )
        cleaned = DocumentCleaner.clean_text(raw_text)
        self.assertNotIn("Dallas ISD rev", cleaned)
        self.assertNotIn("Page 4 of 40", cleaned)
        self.assertIn("Real content line.", cleaned)

    def test_isolated_bullet_repair(self) -> None:
        raw_text = "Header\n\n•\nFirst item\n\n•\nSecond item"
        cleaned = DocumentCleaner.clean_text(raw_text)
        self.assertIn("- First item", cleaned)
        self.assertIn("- Second item", cleaned)

    def test_double_bullet_and_replacement_repair(self) -> None:
        raw_text = (
            "- • Double bullet item\n"
            "* • Star bullet item\n"
            "• Standalone bullet item\n"
            "\ufffd Replacement char item\n"
        )
        cleaned = DocumentCleaner.clean_text(raw_text)
        self.assertIn("- Double bullet item", cleaned)
        self.assertIn("* Star bullet item", cleaned)
        self.assertIn("- Standalone bullet item", cleaned)
        self.assertIn("- Replacement char item", cleaned)

    def test_table_row_deduplication(self) -> None:
        duplicated_row = [
            "Very long compliance question that spans across all columns",
            "Very long compliance question that spans across all columns",
            "Very long compliance question that spans across all columns",
            "Very long compliance question that spans across all columns",
        ]
        deduped = DocumentCleaner.deduplicate_table_row(duplicated_row)
        self.assertEqual(deduped[0], duplicated_row[0])
        self.assertEqual(deduped[1:], ["", "", ""])


class TestHTMLLayoutExtractor(unittest.TestCase):
    """Test suite for HTML table and element parsing into clean Markdown."""

    def test_html_table_parsing_with_colspan(self) -> None:
        html = """
        <table>
            <tr><th>Header 1</th><th>Header 2</th></tr>
            <tr><td colspan="2">Merged Banner Cell Across Columns</td></tr>
            <tr><td>Cell A</td><td>Cell B</td></tr>
        </table>
        """
        md, tables = HTMLLayoutExtractor.html_to_markdown_and_tables(html)
        self.assertEqual(len(tables), 1)
        grid = tables[0]
        self.assertEqual(len(grid), 3)
        self.assertIn("| Header 1 | Header 2 |", md)
        self.assertIn("Merged Banner Cell Across Columns", md)

    def test_html_checkbox_and_list_conversion(self) -> None:
        html = """
        <h1>Form Header</h1>
        <p><input type="checkbox" checked> Option A</p>
        <p><input type="checkbox"> Option B</p>
        <ul>
            <li>Bullet point one</li>
            <li>Bullet point two</li>
        </ul>
        """
        md, _ = HTMLLayoutExtractor.html_to_markdown_and_tables(html)
        self.assertIn("# Form Header", md)
        self.assertIn("[x] Option A", md)
        self.assertIn("[ ] Option B", md)
        self.assertIn("- Bullet point one", md)


class TestPageLayoutRouter(unittest.TestCase):
    """Test suite for PageLayoutRouter metrics and tier routing."""

    def setUp(self) -> None:
        self.router = PageLayoutRouter()

    def test_metrics_extraction_on_real_page(self) -> None:
        doc = pymupdf.open(BID1_PDF)
        with doc:
            page = doc[0]
            metrics = self.router.get_layout_complexity_metrics(page)
            self.assertIsInstance(metrics, LayoutMetrics)
            self.assertGreaterEqual(metrics.text_length, 0)
            self.assertGreaterEqual(metrics.img_coverage, 0.0)

    def test_form_checkbox_routing_to_balanced(self) -> None:
        doc = pymupdf.open(BID1_PDF)
        with doc:
            page_41 = doc[40]
            result = self.router.route_page(page_41, page_number=41)
            self.assertEqual(result.mode, PageProcessingMode.CHANDRA_BALANCED)

    def test_table_page_routing_to_fast(self) -> None:
        doc = pymupdf.open(BID1_PDF)
        with doc:
            page_2 = doc[1]
            result = self.router.route_page(page_2, page_number=2)
            self.assertEqual(result.mode, PageProcessingMode.CHANDRA_FAST)

    def test_vector_sparse_sheet_routing_to_balanced(self) -> None:
        doc = pymupdf.open(BID1_PDF)
        with doc:
            page_55 = doc[54]
            result = self.router.route_page(page_55, page_number=55)
            self.assertEqual(result.mode, PageProcessingMode.CHANDRA_BALANCED)


class TestChandraOCRClient(unittest.TestCase):
    """Test suite for ChandraOCRClient cache and telemetry."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)
        self.client = ChandraOCRClient(api_key="test_dummy_key", cache_dir=self.cache_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_cache_hit_retrieval(self) -> None:
        doc = pymupdf.open(BID2_PDF)
        with doc:
            page = doc[0]
            stream_bytes = page.read_contents() or f"{page.rect}".encode("utf-8")
            cache_key = AsyncChandraOCRClient.compute_cache_key(
                content_bytes=stream_bytes,
                doc_identifier=BID2_PDF.stem,
                page_index=0,
                mode="fast",
                output_format="html",
            )
            html_cache_file = self.cache_dir / f"{cache_key}.html"
            html_cache_file.write_text("<h1>Cached Title</h1><p>Cached text.</p>", encoding="utf-8")

            md_res = self.client.ocr_page(doc, page_index_0based=0, mode="fast")
            self.assertIn("# Cached Title", md_res)
            self.assertIn("Cached text.", md_res)
            self.assertEqual(self.client.metrics.cache_hits, 1)


if __name__ == "__main__":
    unittest.main()
