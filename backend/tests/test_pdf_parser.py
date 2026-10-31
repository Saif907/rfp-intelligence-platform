"""Unit tests for the production PDF parser engine (PDFParser).

Verifies:
1. Parsing of single-page and multi-page bidding documents.
2. Layout recovery on government PDFs with border boxes (Dallas ISD pages 4-6).
3. Banner splitting on composite government forms (Maryland Dell PORFP).
4. Table extraction and physical page pinning.
5. Error handling on non-PDF, corrupted, empty, and non-existent files.
"""

from pathlib import Path
import sys
import tempfile
import unittest

# Ensure 'backend' directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.logging import setup_logging
from ingestion.exceptions import (
    CorruptedDocumentError,
    EmptyDocumentError,
    UnsupportedFormatError,
)
from ingestion.models import (
    DiscoveredDocument,
    DocumentType,
    FileType,
    ParsedDocument,
)
from ingestion.pdf_parser import PDFParser


class TestPDFParser(unittest.TestCase):
    """Comprehensive test suite for PDF parsing and extraction quality."""

    @classmethod
    def setUpClass(cls) -> None:
        setup_logging(level="WARNING")
        current = Path.cwd().resolve()
        candidates = [current, current.parent]
        if len(current.parents) > 1:
            candidates.append(current.parents[1])

        cls.data_dir = None
        for base in candidates:
            cand = base / "data"
            if cand.exists() and (cand / "Bid1").exists():
                cls.data_dir = cand
                break

        if not cls.data_dir:
            raise FileNotFoundError("Could not find 'data/Bid1' directory for test execution.")

        cls.bid1_dir = cls.data_dir / "Bid1"
        cls.bid2_dir = cls.data_dir / "Bid2"

    def test_parse_single_page_addendum(self) -> None:
        """Verify single-page addendum parsing and content verification."""
        addendum_path = self.bid1_dir / "Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf"
        self.assertTrue(addendum_path.exists(), f"Missing file: {addendum_path}")

        doc = PDFParser.parse_file(
            file_path=addendum_path,
            bid_id="Bid1",
            doc_type=DocumentType.ADDENDUM,
            addendum_number=2,
        )

        self.assertIsInstance(doc, ParsedDocument)
        self.assertEqual(doc.total_pages, 1)
        self.assertEqual(doc.addendum_number, 2)
        self.assertEqual(len(doc.pages), 1)
        self.assertGreater(doc.total_characters, 500)

        page1 = doc.pages[0]
        self.assertEqual(page1.page_number, 1)
        # Addendum 2 extends the deadline to July 9, 2024
        has_extended_deadline = (
            "July 9, 2024" in page1.markdown
            or "07/09/2024" in page1.markdown
            or "July 9" in page1.markdown
        )
        self.assertTrue(has_extended_deadline, "Addendum 2 did not contain extended deadline 'July 9'")

    def test_layout_recovery_on_border_boxes(self) -> None:
        """Verify layout recovery triggers on border-box pages (Dallas ISD pages 4-6)."""
        rfp_path = self.bid1_dir / "JA-207652 Student and Staff Computing Devices FINAL.pdf"
        self.assertTrue(rfp_path.exists(), f"Missing file: {rfp_path}")

        # Parse first 7 pages for fast, deterministic verification
        doc = PDFParser.parse_file(
            file_path=rfp_path,
            bid_id="Bid1",
            doc_type=DocumentType.RFP,
            max_pages=7,
        )

        self.assertEqual(doc.total_pages, 7)
        self.assertGreaterEqual(doc.recovered_pages_count, 3)

        # Page 4 contains hardware specifications recovered from the outer border box
        p4 = doc.pages[3]
        self.assertTrue(p4.recovery_applied)
        self.assertGreater(p4.char_count, 1500)
        self.assertIn("Chromebook", p4.markdown)
        self.assertIn("Minimum Requirements", p4.markdown)

        # Page 5 contains RAM and WiFi specifications
        p5 = doc.pages[4]
        self.assertTrue(p5.recovery_applied)
        has_memory = "Memory" in p5.markdown or "RAM" in p5.markdown
        has_wifi = "Wireless" in p5.markdown or "WiFi" in p5.markdown
        self.assertTrue(has_memory)
        self.assertTrue(has_wifi)

    def test_composite_table_banner_splitting(self) -> None:
        """Verify composite table splitting on Maryland Dell PORFP page 3."""
        porfp_path = self.bid2_dir / "PORFP_-_Dell_Laptop_Final.pdf"
        self.assertTrue(porfp_path.exists(), f"Missing file: {porfp_path}")

        doc = PDFParser.parse_file(
            file_path=porfp_path,
            bid_id="Bid2",
            doc_type=DocumentType.RFP,
        )

        self.assertEqual(doc.total_pages, 4)
        self.assertGreaterEqual(doc.total_tables, 5)

        # Page 3 has the stacked tables
        p3 = doc.pages[2]
        self.assertTrue(p3.has_tables)
        self.assertGreaterEqual(len(p3.tables), 4)

        all_table_text = " ".join(
            cell
            for table in p3.tables
            for row in table
            for cell in row
        )
        self.assertIn("Dell Latitude 5550", all_table_text)
        self.assertIn("SI# CC7802", all_table_text)
        self.assertIn("WD22TB4", all_table_text)

    def test_specs_pdf_parsing(self) -> None:
        """Verify technical specification PDF parsing and headings."""
        specs_path = self.bid2_dir / "Dell_Laptop_Specs.pdf"
        self.assertTrue(specs_path.exists(), f"Missing file: {specs_path}")

        doc = PDFParser.parse_file(
            file_path=specs_path,
            bid_id="Bid2",
            doc_type=DocumentType.SPECS,
        )

        self.assertEqual(doc.total_pages, 3)
        self.assertGreater(doc.total_characters, 4000)
        self.assertEqual(doc.scanned_pages_count, 0)

    def test_parse_discovered_document(self) -> None:
        """Verify parsing when passing a DiscoveredDocument instance."""
        affidavit_path = self.bid2_dir / "Mercury_Affidavit.pdf"
        disc_doc = DiscoveredDocument(
            bid_id="Bid2",
            file_name=affidavit_path.name,
            file_path=affidavit_path,
            file_type=FileType.PDF,
            doc_type=DocumentType.AFFIDAVIT,
            page_count=1,
            file_size_bytes=affidavit_path.stat().st_size,
            classification_reason="Test discovered doc",
        )

        parsed = PDFParser.parse_document(disc_doc)
        self.assertEqual(parsed.bid_id, "Bid2")
        self.assertEqual(parsed.doc_type, DocumentType.AFFIDAVIT)
        self.assertEqual(parsed.total_pages, 1)
        self.assertGreater(parsed.total_characters, 1000)

    def test_parse_non_pdf_raises_error(self) -> None:
        """Verify that passing an HTML or non-PDF file raises UnsupportedFormatError."""
        html_files = list(self.bid1_dir.glob("*.html"))
        self.assertTrue(bool(html_files), "No HTML file found in Bid1 directory")
        html_path = html_files[0]
        disc_doc = DiscoveredDocument(
            bid_id="Bid1",
            file_name=html_path.name,
            file_path=html_path,
            file_type=FileType.HTML,
            doc_type=DocumentType.BID_PAGE,
            page_count=1,
            file_size_bytes=html_path.stat().st_size,
            classification_reason="HTML portal page",
        )

        with self.assertRaises(UnsupportedFormatError):
            PDFParser.parse_document(disc_doc)

    def test_parse_non_existent_file(self) -> None:
        """Verify that parsing a non-existent file path raises FileNotFoundError."""
        missing_path = self.bid1_dir / "NonExistentFile.pdf"
        with self.assertRaises(FileNotFoundError):
            PDFParser.parse_file(missing_path)

    def test_parse_empty_file(self) -> None:
        """Verify that an empty 0-byte file raises CorruptedDocumentError."""
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(b"")
            tmp.flush()
            with self.assertRaises(CorruptedDocumentError):
                PDFParser.parse_file(Path(tmp.name))


if __name__ == "__main__":
    unittest.main()
