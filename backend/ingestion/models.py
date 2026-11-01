"""Metadata schemas and data models for document discovery and classification.

Enforces strict Pydantic v2 contracts for all ingestion stages.
"""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DocumentType(StrEnum):
    """Core document roles required by RFP intelligence platform."""

    BID_PAGE = "bid_page"  # Portal summary/overview HTML page
    RFP = "rfp"  # Main solicitation / RFP / PORFP document
    ADDENDUM = "addendum"  # Formally numbered addendum/amendment modifying terms or dates
    SPECS = "specs"  # Technical hardware/software specification sheets
    AFFIDAVIT = "affidavit"  # Legal affidavits, certifications, compliance forms
    OTHER = "other"  # Supplementary attachments and general documentation


class FileType(StrEnum):
    """Supported physical file formats and failure fallback state."""

    PDF = "pdf"
    HTML = "html"
    UNKNOWN_FORMAT = "unknown_format"  # Preserved for investigation rather than dropped


class DiscoveredDocument(BaseModel):
    """Validated structural metadata for any discovered bidding document.

    Attributes:
        bid_id: Unique identifier for the bid folder or solicitation.
        file_name: Base file name with extension.
        file_path: Absolute resolved path to the file on disk.
        file_type: Validated MIME/file format (PDF, HTML, or UNKNOWN_FORMAT).
        doc_type: Classified procurement role.
        addendum_number: Extracted 1-indexed sequential integer if doc_type is ADDENDUM.
        page_count: Total pages (defaults to 1 for HTML documents).
        file_size_bytes: Size of the file in bytes.
        classification_reason: Explainability audit trail documenting how doc_type was assigned.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bid_id: str = Field(..., description="Unique identifier for the bid folder")
    file_name: str = Field(..., description="Base filename with extension")
    file_path: Path = Field(..., description="Absolute resolved filesystem path")
    file_type: FileType = Field(..., description="File MIME category or UNKNOWN_FORMAT")
    doc_type: DocumentType = Field(..., description="Classified RFP document role")
    addendum_number: int | None = Field(
        default=None,
        ge=1,
        description="Sequential integer if document is an addendum/amendment",
    )
    page_count: int = Field(
        default=1,
        ge=0,
        description="Total page count (0 if corrupted or unreadable)",
    )
    file_size_bytes: int = Field(
        ...,
        ge=0,
        description="Filesystem size in bytes",
    )
    classification_reason: str = Field(
        ...,
        description="Audit reason detailing how the document type or format failure was determined",
    )


class ParsedPage(BaseModel):
    """Normalized page representation with preserved layout and table grids."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: int = Field(..., ge=1, description="1-indexed physical PDF page number")
    markdown: str = Field(..., description="Font- and layout-aware markdown representation")
    char_count: int = Field(..., ge=0, description="Total characters in markdown text")
    headings: list[str] = Field(default_factory=list, description="Extracted section titles")
    tables: list[list[list[str]]] = Field(
        default_factory=list,
        description="Structured table grids (2D matrices of string cells)",
    )
    has_tables: bool = Field(default=False, description="True if one or more valid tables exist")
    is_scanned: bool = Field(
        default=False,
        description="True if page contains <30 characters of readable text",
    )
    recovery_applied: bool = Field(
        default=False,
        description="True if PyMuPDF block layout recovery was triggered to prevent data loss",
    )
    ocr_applied: bool = Field(
        default=False,
        description="True if Chandra 2 OCR was executed to extract layout and text",
    )
    extraction_method: str = Field(
        default="digital",
        description="Extraction provenance: 'digital', 'recovered_blocks', 'chandra_ocr', or 'fallback'",
    )


class ParsedDocument(BaseModel):
    """Full parsed document containing normalized pages and performance metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bid_id: str = Field(..., description="Unique identifier for the bid folder")
    file_name: str = Field(..., description="Base filename with extension")
    file_path: Path = Field(..., description="Absolute resolved filesystem path")
    doc_type: DocumentType = Field(..., description="Classified RFP document role")
    addendum_number: int | None = Field(
        default=None,
        ge=1,
        description="Sequential integer if document is an addendum/amendment",
    )
    total_pages: int = Field(..., ge=1, description="Total number of parsed pages")
    pages: list[ParsedPage] = Field(..., description="List of parsed page objects")
    total_characters: int = Field(..., ge=0, description="Total text characters across all pages")
    total_tables: int = Field(..., ge=0, description="Total structured tables extracted")
    total_headings: int = Field(..., ge=0, description="Total markdown section headings")
    scanned_pages_count: int = Field(..., ge=0, description="Number of pages flagged as scanned/image")
    recovered_pages_count: int = Field(
        ...,
        ge=0,
        description="Number of pages where border-box layout recovery was applied",
    )
    ocr_pages_count: int = Field(
        default=0,
        ge=0,
        description="Number of pages processed with Chandra 2 OCR",
    )
    parse_duration_seconds: float = Field(
        ...,
        ge=0.0,
        description="Execution time in seconds to parse the document",
    )

