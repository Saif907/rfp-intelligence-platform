"""Ingestion module for RFP document discovery, parsing, and normalization."""

from ingestion.classifier import DocumentClassifier
from ingestion.exceptions import (
    CorruptedDocumentError,
    DocumentDiscoveryError,
    EmptyDocumentError,
    IngestionError,
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
from ingestion.pdf_parser import PDFParser

__all__ = [
    "CorruptedDocumentError",
    "DiscoveredDocument",
    "DocumentClassifier",
    "DocumentDiscoveryError",
    "DocumentType",
    "EmptyDocumentError",
    "FileType",
    "IngestionError",
    "PDFParser",
    "PDFParsingError",
    "ParsedDocument",
    "ParsedPage",
    "UnsupportedFormatError",
]
