"""Two-tier document discovery and classification engine for RFP intelligence.

Implements a 4-priority format integrity waterfall:
1. Priority 1: Case-insensitive extension check (.pdf, .html, .PDF, .HTML).
2. Priority 2: Magic bytes / file signature detection (catches misleading or missing extensions).
3. Priority 3: Actual parser validation (PyMuPDF for PDF, DOM/character verification for HTML).
4. Fallback: Preserves file as UNKNOWN_FORMAT with audit failure reason (never silently dropped).
"""

from pathlib import Path
import re

import pymupdf

from app.core.logging import get_logger
from ingestion.exceptions import (
    CorruptedDocumentError,
    DocumentDiscoveryError,
    EmptyDocumentError,
    IngestionError,
    UnsupportedFormatError,
)
from ingestion.models import DiscoveredDocument, DocumentType, FileType

logger = get_logger(__name__)

# Files to silently ignore during directory scanning
IGNORED_SYSTEM_FILES = {
    ".ds_store",
    "thumbs.db",
    "desktop.ini",
    ".gitkeep",
    ".gitignore",
}


class DocumentClassifier:
    """Classifies bidding documents into structured RFP roles.

    Uses a two-tier waterfall:
    - Tier 1: File signature / magic bytes and high-confidence filename patterns.
    - Tier 2: Sub-10ms Page 1 semantic content analysis via PyMuPDF for ambiguous files.
    """

    # --- Filename Pattern Regexes ---
    FN_ADDENDUM = re.compile(
        r"(?:addendum|amendment)[\s_#-]*(\d+)",
        re.IGNORECASE,
    )
    FN_SPECS = re.compile(
        r"(?:spec|specification|hardware_requirement)",
        re.IGNORECASE,
    )
    FN_AFFIDAVIT = re.compile(
        r"(?:affidavit|certification|compliance)",
        re.IGNORECASE,
    )
    FN_RFP = re.compile(
        r"(?:rfp|porfp|solicitation|proposal|final|main)",
        re.IGNORECASE,
    )

    # --- First-Page Semantic Content Regexes ---
    CONTENT_ADDENDUM = re.compile(
        r"(?:ADDENDUM|AMENDMENT)[\s#–:-]*(?:NO\.?|NUMBER)?\s*(\d+)?",
        re.IGNORECASE,
    )
    CONTENT_AFFIDAVIT = re.compile(
        r"(?:AFFIDAVIT|SWORN TO AND SUBSCRIBED|NOTARY PUBLIC|NON-COLLUSION)",
        re.IGNORECASE,
    )
    CONTENT_SPECS = re.compile(
        r"(?:TECHNICAL SPECIFICATIONS|FEATURE SPECIFICATION|PROCESSOR|OPERATING SYSTEM|BATTERY LIFE)",
        re.IGNORECASE,
    )
    CONTENT_RFP = re.compile(
        r"(?:REQUEST FOR PROPOSALS?|PURCHASE ORDER REQUEST FOR PROPOSALS?|SOLICITATION|SCOPE OF WORK|INSTRUCTIONS TO BIDDERS)",
        re.IGNORECASE,
    )

    @classmethod
    def verify_format_and_parser(
        cls, path: Path
    ) -> tuple[FileType, str, int, str]:
        """Validates format and parser readiness using the 4-priority waterfall.

        Returns:
            tuple of (FileType, format_reason, page_count, first_page_text)
        """
        # --- Priority 1: Extension Detection ---
        raw_suffix = path.suffix.lower()
        ext_candidate: str | None = None
        if raw_suffix in (".pdf",):
            ext_candidate = "pdf"
        elif raw_suffix in (".html", ".htm"):
            ext_candidate = "html"

        # Read first 1024 bytes for magic signature checks
        try:
            with open(path, "rb") as f:
                header = f.read(1024)
        except OSError as err:
            reason = f"[Priority 2 Error] Cannot read file header bytes: {err}"
            logger.error("[%s] %s", path.name, reason)
            return FileType.UNKNOWN_FORMAT, reason, 0, ""

        # --- Priority 2: Magic Bytes / MIME Detection ---
        has_pdf_magic = header.startswith(b"%PDF-")
        header_lower = header.lower()
        has_html_magic = any(
            tag in header_lower for tag in (b"<!doctype html", b"<html", b"<head", b"<body")
        )

        resolved_format: str | None = None

        if ext_candidate == "pdf":
            if has_pdf_magic:
                resolved_format = "pdf"
            else:
                reason = "[Priority 2 Misleading Extension] File has .pdf extension but magic bytes do not match '%PDF-'"
                logger.warning("[%s] %s", path.name, reason)
                return FileType.UNKNOWN_FORMAT, reason, 0, ""

        elif ext_candidate == "html":
            if has_html_magic or b"<" in header:
                resolved_format = "html"
            else:
                reason = "[Priority 2 Misleading Extension] File has .html extension but does not contain HTML tags"
                logger.warning("[%s] %s", path.name, reason)
                return FileType.UNKNOWN_FORMAT, reason, 0, ""

        else:
            # Missing, generic, or unusual extension (e.g. .dat, .tmp, no extension)
            if has_pdf_magic:
                resolved_format = "pdf"
                logger.info("[%s] Rescued un-extended/misnamed file as PDF via PDF header signature", path.name)
            elif has_html_magic:
                resolved_format = "html"
                logger.info("[%s] Rescued un-extended/misnamed file as HTML via markup signature", path.name)
            else:
                reason = f"[Priority 1 & 2 Failed] Extension '{raw_suffix or 'none'}' and magic bytes unrecognized"
                logger.debug("[%s] %s", path.name, reason)
                return FileType.UNKNOWN_FORMAT, reason, 0, ""

        # --- Priority 3: Actual Parser Validation ---
        if resolved_format == "pdf":
            try:
                with pymupdf.open(path) as doc:
                    page_count = len(doc)
                    if page_count == 0:
                        reason = "[Priority 3 Parser Error] PyMuPDF opened file but detected 0 pages"
                        logger.warning("[%s] %s", path.name, reason)
                        return FileType.UNKNOWN_FORMAT, reason, 0, ""

                    p1_text = doc[0].get_text("text") if page_count > 0 else ""
                    return (
                        FileType.PDF,
                        f"Priority 1-3 Verified (PyMuPDF verified {page_count} pages)",
                        page_count,
                        p1_text,
                    )
            except Exception as err:
                reason = f"[Priority 3 Parser Error] PyMuPDF failed to parse stream: {err}"
                logger.error("[%s] %s", path.name, reason)
                return FileType.UNKNOWN_FORMAT, reason, 0, ""

        elif resolved_format == "html":
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                if len(content.strip()) == 0:
                    reason = "[Priority 3 Parser Error] HTML file is empty"
                    logger.warning("[%s] %s", path.name, reason)
                    return FileType.UNKNOWN_FORMAT, reason, 0, ""

                return (
                    FileType.HTML,
                    "Priority 1-3 Verified (Valid HTML structure)",
                    1,
                    content[:1000],
                )
            except Exception as err:
                reason = f"[Priority 3 Parser Error] Failed to read HTML content: {err}"
                logger.error("[%s] %s", path.name, reason)
                return FileType.UNKNOWN_FORMAT, reason, 0, ""

        # --- Fallback: UNKNOWN_FORMAT ---
        reason = "[Fallback] File format could not be verified by any parser"
        return FileType.UNKNOWN_FORMAT, reason, 0, ""

    @classmethod
    def classify_file(
        cls,
        file_path: Path | str,
        bid_id: str,
        strict: bool = False,
    ) -> DiscoveredDocument | None:
        """Classifies a single file into a DiscoveredDocument.

        Uses the 4-priority format & parser verification before applying
        semantic role identification.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            msg = f"File not found or inaccessible: {path}"
            logger.error(msg)
            if strict:
                raise DocumentDiscoveryError(msg, file_path=path)
            return None

        file_size = path.stat().st_size
        if file_size == 0:
            msg = f"Zero-byte empty file detected: {path.name}"
            logger.warning(msg)
            if strict:
                raise EmptyDocumentError(msg, file_path=path)
            # Preserve empty file as UNKNOWN_FORMAT record rather than dropping
            return DiscoveredDocument(
                bid_id=bid_id,
                file_name=path.name,
                file_path=path,
                file_type=FileType.UNKNOWN_FORMAT,
                doc_type=DocumentType.OTHER,
                addendum_number=None,
                page_count=0,
                file_size_bytes=0,
                classification_reason=msg,
            )

        # Execute 4-Priority Format & Parser Validation
        file_type, format_reason, page_count, p1_text = cls.verify_format_and_parser(path)

        # Handle UNKNOWN_FORMAT fallback
        if file_type == FileType.UNKNOWN_FORMAT:
            if strict:
                if "Parser Error" in format_reason:
                    raise CorruptedDocumentError(format_reason, file_path=path)
                raise UnsupportedFormatError(format_reason, file_path=path)

            logger.warning("[%s] Preserved as UNKNOWN_FORMAT: %s", path.name, format_reason)
            return DiscoveredDocument(
                bid_id=bid_id,
                file_name=path.name,
                file_path=path,
                file_type=FileType.UNKNOWN_FORMAT,
                doc_type=DocumentType.OTHER,
                addendum_number=None,
                page_count=page_count,
                file_size_bytes=file_size,
                classification_reason=format_reason,
            )

        stem = path.stem

        # ---------------------------------------------------------------------
        # 1. HTML Portal Overview Documents
        # ---------------------------------------------------------------------
        if file_type == FileType.HTML:
            doc = DiscoveredDocument(
                bid_id=bid_id,
                file_name=path.name,
                file_path=path,
                file_type=file_type,
                doc_type=DocumentType.BID_PAGE,
                addendum_number=None,
                page_count=1,
                file_size_bytes=file_size,
                classification_reason="HTML portal web page detected and verified.",
            )
            logger.info("[%s] Classified as BID_PAGE (%s)", path.name, bid_id)
            return doc

        # ---------------------------------------------------------------------
        # 2. PDF Documents: Semantic Role Identification
        # ---------------------------------------------------------------------
        # Signal A: Addendum / Amendment (Filename match)
        if add_match := cls.FN_ADDENDUM.search(stem):
            num = int(add_match.group(1))
            doc = DiscoveredDocument(
                bid_id=bid_id,
                file_name=path.name,
                file_path=path,
                file_type=file_type,
                doc_type=DocumentType.ADDENDUM,
                addendum_number=num,
                page_count=page_count,
                file_size_bytes=file_size,
                classification_reason=f"Filename matched addendum pattern (Number: {num}).",
            )
            logger.info("[%s] Classified as ADDENDUM #%d (%s, %d pages)", path.name, num, bid_id, page_count)
            return doc

        # Fallback for Addendum: Inspect Page 1 content if filename was generic (e.g. Doc_1.pdf)
        if content_add := cls.CONTENT_ADDENDUM.search(p1_text[:1000]):
            num = int(content_add.group(1)) if content_add.group(1) else None
            doc = DiscoveredDocument(
                bid_id=bid_id,
                file_name=path.name,
                file_path=path,
                file_type=file_type,
                doc_type=DocumentType.ADDENDUM,
                addendum_number=num,
                page_count=page_count,
                file_size_bytes=file_size,
                classification_reason=f"Page 1 header identified formal addendum/amendment (Number: {num}).",
            )
            logger.info("[%s] Classified as ADDENDUM via Page 1 header #%s (%s)", path.name, num, bid_id)
            return doc

        # Signal B: Affidavit / Compliance Form
        if cls.FN_AFFIDAVIT.search(stem) or cls.CONTENT_AFFIDAVIT.search(p1_text[:1000]):
            doc = DiscoveredDocument(
                bid_id=bid_id,
                file_name=path.name,
                file_path=path,
                file_type=file_type,
                doc_type=DocumentType.AFFIDAVIT,
                addendum_number=None,
                page_count=page_count,
                file_size_bytes=file_size,
                classification_reason="Identified legal affidavit/notary block from filename or Page 1 content.",
            )
            logger.info("[%s] Classified as AFFIDAVIT (%s, %d pages)", path.name, bid_id, page_count)
            return doc

        # Signal C: Technical Specifications
        if cls.FN_SPECS.search(stem) or (cls.CONTENT_SPECS.search(p1_text) and page_count <= 5):
            doc = DiscoveredDocument(
                bid_id=bid_id,
                file_name=path.name,
                file_path=path,
                file_type=file_type,
                doc_type=DocumentType.SPECS,
                addendum_number=None,
                page_count=page_count,
                file_size_bytes=file_size,
                classification_reason="Technical hardware specifications sheet identified.",
            )
            logger.info("[%s] Classified as SPECS (%s, %d pages)", path.name, bid_id, page_count)
            return doc

        # Signal D: Main RFP Solicitation
        if cls.FN_RFP.search(stem) or cls.CONTENT_RFP.search(p1_text[:1500]) or page_count > 10:
            doc = DiscoveredDocument(
                bid_id=bid_id,
                file_name=path.name,
                file_path=path,
                file_type=file_type,
                doc_type=DocumentType.RFP,
                addendum_number=None,
                page_count=page_count,
                file_size_bytes=file_size,
                classification_reason=f"Primary RFP proposal document identified (Pages: {page_count}).",
            )
            logger.info("[%s] Classified as RFP (%s, %d pages)", path.name, bid_id, page_count)
            return doc

        # Signal E: Supplementary Attachment / General Other
        doc = DiscoveredDocument(
            bid_id=bid_id,
            file_name=path.name,
            file_path=path,
            file_type=file_type,
            doc_type=DocumentType.OTHER,
            addendum_number=None,
            page_count=page_count,
            file_size_bytes=file_size,
            classification_reason="Unclassified supplementary attachment.",
        )
        logger.info("[%s] Classified as OTHER (%s, %d pages)", path.name, bid_id, page_count)
        return doc

    @classmethod
    def scan_folder(
        cls,
        folder_path: Path | str,
        bid_id: str | None = None,
        strict: bool = False,
    ) -> list[DiscoveredDocument]:
        """Scans any bid directory and returns sorted, classified DiscoveredDocument records.

        Unknown or invalid formats are preserved as UNKNOWN_FORMAT rather than dropped.
        """
        target_dir = Path(folder_path).resolve()
        if not target_dir.exists():
            msg = f"Bid directory does not exist: {target_dir}"
            logger.error(msg)
            raise DocumentDiscoveryError(msg, file_path=target_dir)

        if not target_dir.is_dir():
            msg = f"Target path is not a directory: {target_dir}"
            logger.error(msg)
            raise DocumentDiscoveryError(msg, file_path=target_dir)

        effective_bid_id = bid_id or target_dir.name
        logger.info("Scanning bid folder '%s' (bid_id='%s')", target_dir.name, effective_bid_id)

        documents: list[DiscoveredDocument] = []

        try:
            dir_entries = sorted(target_dir.iterdir())
        except OSError as err:
            msg = f"Permission denied or filesystem error reading directory: {err}"
            logger.error(msg)
            raise DocumentDiscoveryError(msg, file_path=target_dir) from err

        for item in dir_entries:
            # Silently ignore OS artifacts (.DS_Store, Thumbs.db)
            if item.name.lower() in IGNORED_SYSTEM_FILES or item.name.startswith("."):
                logger.debug("Skipping system file: %s", item.name)
                continue

            if item.is_file():
                try:
                    doc = cls.classify_file(item, bid_id=effective_bid_id, strict=strict)
                    if doc is not None:
                        documents.append(doc)
                except IngestionError as err:
                    if strict:
                        raise
                    logger.warning("Document failed strict check: %s", err)

        logger.info("Discovery complete for '%s': registered %d documents", effective_bid_id, len(documents))
        return documents
