"""Unit tests for the unified IngestionPipeline."""

import json
import tempfile
import unittest
from pathlib import Path

from ingestion.exceptions import UnsupportedFormatError
from ingestion.models import DocumentType, FileType, ParsedDocument, ParsedPage
from ingestion.pipeline import IngestionPipeline


class TestIngestionPipeline(unittest.TestCase):
    """Test suite covering IngestionPipeline orchestration, parsing, and export."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace_root = Path(__file__).resolve().parent.parent.parent
        cls.data_dir = cls.workspace_root / "data"
        cls.bid1_dir = cls.data_dir / "Bid1"
        cls.bid2_dir = cls.data_dir / "Bid2"

    def setUp(self) -> None:
        self.pipeline = IngestionPipeline(enable_ocr=True, max_pages=3)

    def test_pipeline_initialization(self) -> None:
        """Verifies pipeline config options."""
        p = IngestionPipeline(enable_ocr=False, max_pages=5)
        self.assertFalse(p.enable_ocr)
        self.assertEqual(p.max_pages, 5)

    def test_parse_single_pdf(self) -> None:
        """Verifies parsing of a single PDF via pipeline."""
        target_pdf = self.bid2_dir / "Mercury_Affidavit.pdf"
        if not target_pdf.exists():
            self.skipTest(f"File not found: {target_pdf}")

        parsed = self.pipeline.parse_single_file(target_pdf, bid_id="Bid2")
        self.assertIsInstance(parsed, ParsedDocument)
        self.assertEqual(parsed.bid_id, "Bid2")
        self.assertEqual(parsed.file_name, "Mercury_Affidavit.pdf")
        self.assertEqual(parsed.doc_type, DocumentType.AFFIDAVIT)
        self.assertEqual(len(parsed.pages), 1)

    def test_parse_single_html(self) -> None:
        """Verifies parsing of a single HTML portal page via pipeline."""
        html_files = list(self.bid1_dir.glob("*.html"))
        if not html_files:
            self.skipTest(f"No HTML file found in {self.bid1_dir}")

        target_html = html_files[0]
        parsed = self.pipeline.parse_single_file(target_html, bid_id="Bid1")
        self.assertIsInstance(parsed, ParsedDocument)
        self.assertEqual(parsed.bid_id, "Bid1")
        self.assertEqual(parsed.doc_type, DocumentType.BID_PAGE)
        self.assertGreater(parsed.total_characters, 100)

    def test_unsupported_format_raises(self) -> None:
        """Verifies that attempting to parse an unsupported format raises an error."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmp:
            tmp.write(b"random binary data")
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(UnsupportedFormatError):
                self.pipeline.parse_single_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_process_bid_folder_ordering(self) -> None:
        """Verifies that documents in a bid folder are processed in procurement order."""
        docs = self.pipeline.process_bid_folder(self.bid2_dir, bid_id="Bid2")
        self.assertGreaterEqual(len(docs), 3)

        # Confirm first document is RFP
        self.assertEqual(docs[0].doc_type, DocumentType.RFP)

        # Confirm all documents have valid pages
        for doc in docs:
            self.assertGreater(len(doc.pages), 0)
            self.assertGreater(doc.total_characters, 0)

    def test_export_artifacts_and_manifest(self) -> None:
        """Verifies that artifacts and manifest JSON are properly serialized and validated."""
        with tempfile.TemporaryDirectory() as tmp_out:
            # Sample parse of bid2
            docs = self.pipeline.process_bid_folder(self.bid2_dir, bid_id="Bid2")
            manifest_path = self.pipeline.export_artifacts(
                docs, output_dir=tmp_out, export_json=True
            )

            self.assertTrue(manifest_path.exists())
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest_data["total_documents"], len(docs))
            self.assertIn("Bid2", manifest_data["bids"])
            self.assertGreater(manifest_data["aggregate_statistics"]["total_characters"], 0)

            # Verify individual files exist
            for doc in docs:
                stem = Path(doc.file_name).stem
                json_path = Path(tmp_out) / doc.bid_id / f"{stem}.json"
                md_path = Path(tmp_out) / doc.bid_id / f"{stem}.md"
                self.assertTrue(json_path.exists(), f"JSON file missing: {json_path}")
                self.assertTrue(md_path.exists(), f"Markdown file missing: {md_path}")

                # Validate JSON schema matches ParsedDocument
                parsed_back = ParsedDocument.model_validate_json(json_path.read_text(encoding="utf-8"))
                self.assertEqual(parsed_back.file_name, doc.file_name)


if __name__ == "__main__":
    unittest.main()

