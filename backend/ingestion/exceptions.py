"""Custom exceptions hierarchy for the ingestion pipeline.

Provides structured, specific error types so the orchestrator and agents
can handle parsing failures, unreadable files, and format issues gracefully.
"""

from pathlib import Path


class IngestionError(Exception):
    """Base exception for all document ingestion and parsing errors."""

    def __init__(self, message: str, file_path: Path | str | None = None) -> None:
        self.file_path = Path(file_path) if file_path else None
        self.message = message
        super().__init__(f"[{self.file_path.name if self.file_path else 'General'}] {message}")


class DocumentDiscoveryError(IngestionError):
    """Raised when a bid folder cannot be scanned or accessed."""


class CorruptedDocumentError(IngestionError):
    """Raised when a file has broken magic bytes or cannot be parsed by PyMuPDF/BeautifulSoup."""


class EmptyDocumentError(IngestionError):
    """Raised when a document is 0 bytes or contains 0 readable pages/characters."""


class UnsupportedFormatError(IngestionError):
    """Raised when encountering an unsupported file format that cannot be ingested."""


class PDFParsingError(IngestionError):
    """Raised when a PDF fails to parse during layout or table extraction."""


class HTMLParsingError(IngestionError):
    """Raised when an HTML file fails to parse during DOM or table extraction."""

