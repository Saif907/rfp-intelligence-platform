"""Ingestion module for RFP document discovery, parsing, and normalization."""

from ingestion.classifier import DocumentClassifier
from ingestion.exceptions import (
    CorruptedDocumentError,
    DocumentDiscoveryError,
    EmptyDocumentError,
    HTMLParsingError,
    IngestionError,
    PDFParsingError,
    UnsupportedFormatError,
)
from ingestion.html_parser import HTMLParser
from ingestion.models import (
    DiscoveredDocument,
    DocumentType,
    FileType,
    ParsedDocument,
    ParsedPage,
)
from ingestion.ocr import ChandraOCRClient
from ingestion.pdf_parser import PDFParser
from ingestion.pipeline import IngestionPipeline

__all__ = [
    "ChandraOCRClient",
    "CorruptedDocumentError",
    "DiscoveredDocument",
    "DocumentClassifier",
    "DocumentDiscoveryError",
    "DocumentType",
    "EmptyDocumentError",
    "FileType",
    "HTMLParser",
    "HTMLParsingError",
    "IngestionError",
    "IngestionPipeline",
    "PDFParser",
    "PDFParsingError",
    "ParsedDocument",
    "ParsedPage",
    "UnsupportedFormatError",
]
