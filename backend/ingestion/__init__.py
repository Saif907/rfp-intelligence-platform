"""Ingestion module for RFP document discovery, parsing, and normalization."""

from ingestion.classifier import DocumentClassifier
from ingestion.cleaner import DocumentCleaner
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
from ingestion.ocr import (
    AsyncChandraOCRClient,
    ChandraOCRClient,
    HTMLLayoutExtractor,
    OCRMonitoringMetrics,
)
from ingestion.pdf_parser import PDFParser
from ingestion.pipeline import IngestionPipeline
from ingestion.router import (
    DocumentRoutingPlan,
    LayoutMetrics,
    PageLayoutRouter,
    PageProcessingMode,
    PageRouteResult,
    default_router,
)

__all__ = [
    "AsyncChandraOCRClient",
    "ChandraOCRClient",
    "CorruptedDocumentError",
    "DiscoveredDocument",
    "DocumentClassifier",
    "DocumentCleaner",
    "DocumentDiscoveryError",
    "DocumentRoutingPlan",
    "DocumentType",
    "EmptyDocumentError",
    "FileType",
    "HTMLLayoutExtractor",
    "HTMLParser",
    "HTMLParsingError",
    "IngestionError",
    "IngestionPipeline",
    "LayoutMetrics",
    "OCRMonitoringMetrics",
    "PDFParser",
    "PDFParsingError",
    "PageLayoutRouter",
    "PageProcessingMode",
    "PageRouteResult",
    "ParsedDocument",
    "ParsedPage",
    "UnsupportedFormatError",
    "default_router",
]
