"""Unit tests for the 4-Priority Format & Parser Validation Waterfall.

Tests:
- Priority 1: Case-insensitive extension detection (.pdf, .PDF, .html, .HTML).
- Priority 2: Magic bytes validation & rescuing unextended files / catching fake extensions.
- Priority 3: Actual parser validation via PyMuPDF / HTML checks.
- Fallback: Preservation of UNKNOWN_FORMAT with audit reasons instead of silent discarding.
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
from ingestion.classifier import DocumentClassifier
from ingestion.exceptions import (
    CorruptedDocumentError,
    DocumentDiscoveryError,
    EmptyDocumentError,
    UnsupportedFormatError,
)
from ingestion.models import DocumentType, FileType


class TestFormatAndParserWaterfall(unittest.TestCase):
    """Test suite covering Priority 1 -> 2 -> 3 -> Fallback."""

    @classmethod
    def setUpClass(cls) -> None:
        setup_logging(level="DEBUG")
        current = Path.cwd().resolve()
        candidates = [current, current.parent]
        if len(current.parents) > 1:
            candidates.append(current.parents[1])

        root = next((p for p in candidates if (p / "data").exists()), current)
        cls.data_dir = root / "data"

    def test_priority_1_case_insensitive_extension(self) -> None:
        """Verifies Priority 1 catches case variations like .PDF and .HTML."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            # Create real minimal PDF with uppercase extension
            pdf_upper = tmp / "UPPERCASE.PDF"
            import pymupdf
            doc = pymupdf.open()
            doc.new_page()
            doc.save(str(pdf_upper))
            doc.close()

            # Create real HTML with uppercase extension
            html_upper = tmp / "PORTAL.HTML"
            html_upper.write_text("<!DOCTYPE html><html><body>Portal</body></html>")

            # Validate both pass Priority 1, 2, and 3
            ft_pdf, reason_pdf, p_pdf, _ = DocumentClassifier.verify_format_and_parser(pdf_upper)
            self.assertEqual(ft_pdf, FileType.PDF)
            self.assertEqual(p_pdf, 1)

            ft_html, reason_html, p_html, _ = DocumentClassifier.verify_format_and_parser(html_upper)
            self.assertEqual(ft_html, FileType.HTML)
            self.assertEqual(p_html, 1)

    def test_priority_2_catches_misleading_extension(self) -> None:
        """Verifies Priority 2 catches a text file deceitfully renamed as .pdf."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            fake_pdf = tmp / "malicious_fake.pdf"
            fake_pdf.write_text("This is just plain text, not a real PDF binary!")

            file_type, reason, page_count, _ = DocumentClassifier.verify_format_and_parser(fake_pdf)

            # Must fail Priority 2 magic bytes and fall back to UNKNOWN_FORMAT
            self.assertEqual(file_type, FileType.UNKNOWN_FORMAT)
            self.assertIn("Priority 2 Misleading Extension", reason)
            self.assertEqual(page_count, 0)

    def test_priority_2_rescues_unextended_pdf(self) -> None:
        """Verifies Priority 2 rescues a file with no extension by inspecting %PDF- magic bytes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            unextended_file = tmp / "document_without_extension"

            # Create real PDF binary
            import pymupdf
            doc = pymupdf.open()
            doc.new_page()
            doc.save(str(unextended_file))
            doc.close()

            # Verify it is rescued as FileType.PDF via magic bytes
            file_type, reason, page_count, _ = DocumentClassifier.verify_format_and_parser(unextended_file)
            self.assertEqual(file_type, FileType.PDF)
            self.assertEqual(page_count, 1)
            self.assertIn("PyMuPDF verified 1 pages", reason)

    def test_priority_3_catches_corrupted_pdf_stream(self) -> None:
        """Verifies Priority 3 catches a file with %PDF- header but damaged/truncated internal streams."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            corrupted_pdf = tmp / "corrupted_stream.pdf"
            corrupted_pdf.write_bytes(b"%PDF-1.7\n%BROKEN_TRUNCATED_BODY_WITH_NO_XREF_OR_TRAILER")

            file_type, reason, page_count, _ = DocumentClassifier.verify_format_and_parser(corrupted_pdf)

            # Must fail Priority 3 parser validation
            self.assertEqual(file_type, FileType.UNKNOWN_FORMAT)
            self.assertIn("Priority 3 Parser Error", reason)
            self.assertEqual(page_count, 0)

    def test_fallback_preserves_unknown_format_without_dropping(self) -> None:
        """Verifies that unknown formats are preserved as DiscoveredDocument records with audit trails."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            zip_file = tmp / "attachment_archive.zip"
            zip_file.write_bytes(b"PK\x03\x04SomeZipArchivePayload")

            # In lenient mode, file is NOT dropped: it is preserved as UNKNOWN_FORMAT
            doc = DocumentClassifier.classify_file(zip_file, bid_id="TestBid", strict=False)

            self.assertIsNotNone(doc, "Unknown format must not be silently dropped!")
            self.assertEqual(doc.file_type, FileType.UNKNOWN_FORMAT)
            self.assertEqual(doc.doc_type, DocumentType.OTHER)
            self.assertIn("Priority 1 & 2 Failed", doc.classification_reason)

    def test_real_bid_files_pass_all_priorities(self) -> None:
        """Verifies that all 9 real files across Bid 1 & Bid 2 pass Priorities 1-3 with 0 UNKNOWN_FORMATs."""
        bid1_docs = DocumentClassifier.scan_folder(self.data_dir / "Bid1", bid_id="Bid1")
        bid2_docs = DocumentClassifier.scan_folder(self.data_dir / "Bid2", bid_id="Bid2")

        all_docs = bid1_docs + bid2_docs
        self.assertEqual(len(all_docs), 9)

        for d in all_docs:
            self.assertNotEqual(
                d.file_type,
                FileType.UNKNOWN_FORMAT,
                f"Real file {d.file_name} unexpectedly failed format validation!",
            )
            self.assertGreaterEqual(d.page_count, 1)


if __name__ == "__main__":
    unittest.main()
