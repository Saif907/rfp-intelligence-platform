"""Unit tests for the HTML parser engine (HTMLParser).

Verifies:
1. Parsing of Bid 1 Dallas ISD HTML portal page.
2. Parsing of Bid 2 Maryland Treasurer HTML portal page.
3. Extraction of structured metadata fields and conversion into Markdown and tables.
4. Generic HTML portal generalization.
5. Error handling on non-HTML, empty, and non-existent files.
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
    EmptyDocumentError,
    UnsupportedFormatError,
)
from ingestion.html_parser import HTMLParser
from ingestion.models import (
    DiscoveredDocument,
    DocumentType,
    FileType,
    ParsedDocument,
)


class TestHTMLParser(unittest.TestCase):
    """Test suite for HTML portal parsing and metadata extraction."""

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

    def test_parse_bid1_html(self) -> None:
        """Verify Bid 1 (Dallas ISD) portal HTML parsing."""
        html_files = list(self.bid1_dir.glob("*.html"))
        self.assertTrue(bool(html_files), "No HTML file found in Bid1 directory")
        html_path = html_files[0]

        doc = HTMLParser.parse_file(
            file_path=html_path,
            bid_id="Bid1",
            doc_type=DocumentType.BID_PAGE,
        )

        self.assertIsInstance(doc, ParsedDocument)
        self.assertEqual(doc.total_pages, 1)
        self.assertEqual(doc.doc_type, DocumentType.BID_PAGE)
        self.assertGreater(doc.total_characters, 1000)
        self.assertGreaterEqual(doc.total_tables, 1)

        p = doc.pages[0]
        self.assertEqual(p.page_number, 1)
        self.assertTrue(p.has_tables)

        # Verify critical procurement values
        self.assertIn("JA-207652", p.markdown)
        self.assertIn("Dallas Independent School District", p.markdown)
        self.assertIn("07/09/2024", p.markdown)
        self.assertIn("Piggyback Contract", p.markdown)

    def test_parse_bid2_html(self) -> None:
        """Verify Bid 2 (Maryland Treasurer) portal HTML parsing."""
        html_files = list(self.bid2_dir.glob("*.html"))
        self.assertTrue(bool(html_files), "No HTML file found in Bid2 directory")
        html_path = html_files[0]

        doc = HTMLParser.parse_file(
            file_path=html_path,
            bid_id="Bid2",
            doc_type=DocumentType.BID_PAGE,
        )

        self.assertIsInstance(doc, ParsedDocument)
        self.assertEqual(doc.total_pages, 1)
        self.assertEqual(doc.doc_type, DocumentType.BID_PAGE)

        p = doc.pages[0]
        # Verify official State of Maryland solicitation number BPM044557
        self.assertIn("BPM044557", p.markdown)
        self.assertIn("Maryland Treasurer", p.markdown)
        self.assertIn("06/10/2024", p.markdown)

    def test_parse_generic_html(self) -> None:
        """Verify parser generalizes to arbitrary procurement HTML tables."""
        sample_html = """
        <!DOCTYPE html>
        <html>
        <head><title>City of Austin - RFP #9988</title></head>
        <body>
            <h1>RFP #9988 - Network Switches</h1>
            <table border="1">
                <tr><th>Field</th><th>Value</th></tr>
                <tr><td>Solicitation Number</td><td>RFP-9988</td></tr>
                <tr><td>Due Date</td><td>2024-12-01</td></tr>
                <tr><td>Budget</td><td>$500,000</td></tr>
            </table>
            <p>Vendors must deliver to 505 Barton Springs Rd.</p>
        </body>
        </html>
        """
        with tempfile.NamedTemporaryFile("w", suffix=".html", encoding="utf-8", delete=False) as tmp:
            tmp.write(sample_html)
            tmp_path = Path(tmp.name)

        try:
            doc = HTMLParser.parse_file(file_path=tmp_path, bid_id="BidGeneric")
            self.assertEqual(doc.total_pages, 1)
            p = doc.pages[0]
            self.assertIn("RFP #9988", p.markdown)
            self.assertTrue(p.has_tables)
            self.assertGreaterEqual(len(p.tables), 1)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_parse_discovered_document(self) -> None:
        """Verify parse_document works with DiscoveredDocument."""
        html_files = list(self.bid1_dir.glob("*.html"))
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

        doc = HTMLParser.parse_document(disc_doc)
        self.assertEqual(doc.bid_id, "Bid1")
        self.assertEqual(doc.doc_type, DocumentType.BID_PAGE)

    def test_parse_non_html_raises_error(self) -> None:
        """Verify passing a non-HTML file raises UnsupportedFormatError."""
        pdf_path = self.bid1_dir / "Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf"
        disc_doc = DiscoveredDocument(
            bid_id="Bid1",
            file_name=pdf_path.name,
            file_path=pdf_path,
            file_type=FileType.PDF,
            doc_type=DocumentType.ADDENDUM,
            page_count=1,
            file_size_bytes=pdf_path.stat().st_size,
            classification_reason="PDF addendum",
        )

        with self.assertRaises(UnsupportedFormatError):
            HTMLParser.parse_document(disc_doc)

    def test_parse_empty_file_raises_error(self) -> None:
        """Verify empty HTML file raises EmptyDocumentError."""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            tmp.write(b"")
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(EmptyDocumentError):
                HTMLParser.parse_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_parse_non_existent_file(self) -> None:
        """Verify non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            HTMLParser.parse_file("does_not_exist.html")


if __name__ == "__main__":
    unittest.main()
