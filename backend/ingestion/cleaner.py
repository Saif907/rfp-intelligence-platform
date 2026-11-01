"""Production layout cleaning and normalization engine for procurement documents.

Cleans and normalizes text and tabular data from both PyMuPDF and OCR engines:
1. Strips repetitive running headers and footers across page boundaries.
2. Normalizes isolated bullet symbols (•, -, \uf0b7) into valid Markdown list items.
3. Collapses duplicated merged table cells (e.g. 5x repeated cell text) into single rows.
4. Trims excessive dot-leader chains (......) and line dividers in questionnaires.
5. Replaces Private Use Area (PUA) font glyphs with standard Unicode equivalents.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentCleaner:
    """Enterprise text and table layout sanitizer for RAG indexing."""

    # Common procurement running headers and footers in Bid documents (generalized for unseen bids)
    RUNNING_HEADER_PATTERNS = [
        re.compile(
            r"^\s*(?:#+\s*)?(?:Request\s+For\s+Proposal|RFP|Solicitation|Invitation\s+(?:for|to)\s+Bid|ITB|IFB)\b[^\n]*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r"^\s*[\w\s\.\,\-]+?\brev(?:ision|\.)?\s+\d+(?:\.\d+)*\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r"^\s*Page\s+\d+\s*(?:of|/)\s*\d+\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r"^\s*(?:#+\s*)?(?:SCOPE\s+AND\s+SPECIFICATIONS\s+OF\s+THE\s+PROPOSAL|SPECIFICATIONS\s+CONT(?:INUED|\.)?)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    ]

    # Isolated bullet markers (e.g. \n•\n or \n\uf0b7\n)
    ISOLATED_BULLET_PATTERN = re.compile(
        r"(?:\n|^)\s*[•\u2022\u25cf\uf0b7\ufffd]\s*\n\s*([^\n\s])",
        re.MULTILINE,
    )

    # Dot-leader chains in questionnaire forms (e.g. ................)
    DOT_LEADER_PATTERN = re.compile(
        r"[.…]{4,}",
    )

    # Excessive horizontal divider dashes (e.g. -----------------------)
    DIVIDER_PATTERN = re.compile(
        r"-{10,}",
    )

    @classmethod
    def clean_text(cls, text: str, strip_headers_footers: bool = True) -> str:
        """Sanitizes text by removing boilerplate noise and repairing formatting."""
        if not text:
            return ""

        cleaned = text

        # 1. Replace Private Use Area (PUA) font symbols and broken unicode replacement chars with standard bullet
        cleaned = re.sub(r"[\uf0b7\uf0a7\uf0d8\ufffd]", "•", cleaned)

        # 2. Repair double bullets: '- • Item' or '* • Item' -> '- Item'
        cleaned = re.sub(
            r"^(\s*[-*]\s*)[•\u2022\u25cf\uf0b7\ufffd]\s*",
            r"\1",
            cleaned,
            flags=re.MULTILINE,
        )

        # 3. Repair line-start bullets: '• Item' -> '- Item'
        cleaned = re.sub(
            r"^\s*[•\u2022\u25cf\uf0b7\ufffd]\s*([^\n\s])",
            r"- \1",
            cleaned,
            flags=re.MULTILINE,
        )

        # 4. Repair multi-line isolated bullets: '\n•\nText' -> '\n- Text'
        cleaned = cls.ISOLATED_BULLET_PATTERN.sub(r"\n- \1", cleaned)

        # 5. Clean standalone orphan bullets on their own line
        cleaned = re.sub(
            r"^\s*[•\u2022\u25cf\uf0b7\ufffd]\s*$",
            "",
            cleaned,
            flags=re.MULTILINE,
        )

        # 6. Clean repetitive dot-leaders and separator dashes in questionnaires
        cleaned = cls.DOT_LEADER_PATTERN.sub("...", cleaned)
        cleaned = cls.DIVIDER_PATTERN.sub("---", cleaned)

        # 7. Strip running headers and footers if requested
        if strip_headers_footers:
            for pattern in cls.RUNNING_HEADER_PATTERNS:
                cleaned = pattern.sub("", cleaned)

        # 8. Condense multiple blank lines to at most two
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    @classmethod
    def deduplicate_table_row(cls, cells: list[str]) -> list[str]:
        """Detects merged-cell duplication across columns and collapses it.

        When complex form tables have a single wide cell spanning multiple columns,
        parsers often duplicate the identical long string into every column.
        This method replaces duplicated spans with empty cells or a unified banner.
        """
        non_empty = [c.strip() for c in cells if c.strip()]
        if not non_empty:
            return cells

        # If 3 or more cells have identical content longer than 20 chars
        if len(non_empty) >= 3 and len(set(non_empty)) == 1 and len(non_empty[0]) > 20:
            # Retain only first cell and leave remaining as empty strings
            banner_val = non_empty[0]
            return [banner_val] + [""] * (len(cells) - 1)

        return cells

    @classmethod
    def clean_markdown_tables(cls, markdown: str) -> str:
        """Identifies markdown tables and collapses duplicated merged-cell rows."""
        lines = markdown.splitlines()
        output_lines: list[str] = []
        table_buffer: list[str] = []

        def flush_table(buf: list[str]) -> list[str]:
            if not buf:
                return []
            if len(buf) < 2:
                return buf

            cleaned_tbl_lines: list[str] = []
            for idx, raw_line in enumerate(buf):
                stripped = raw_line.strip()
                if not (stripped.startswith("|") and stripped.endswith("|")):
                    cleaned_tbl_lines.append(raw_line)
                    continue

                cells = [c.strip() for c in stripped.strip("|").split("|")]
                # Don't touch separator row (e.g. |---|---|)
                if all(re.match(r"^:?-+:?$", c) for c in cells if c):
                    cleaned_tbl_lines.append(raw_line)
                    continue

                # Deduplicate row
                clean_cells = cls.deduplicate_table_row(cells)
                cleaned_tbl_lines.append("| " + " | ".join(clean_cells) + " |")

            return cleaned_tbl_lines

        in_table = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                in_table = True
                table_buffer.append(line)
            else:
                if in_table:
                    output_lines.extend(flush_table(table_buffer))
                    table_buffer = []
                    in_table = False
                output_lines.append(line)

        if table_buffer:
            output_lines.extend(flush_table(table_buffer))

        return "\n".join(output_lines)

    @classmethod
    def clean_page_markdown(cls, markdown: str) -> str:
        """Applies full cleaning pipeline to a page's markdown text."""
        # Clean text artifacts and headers
        cleaned = cls.clean_text(markdown)
        # Clean and deduplicate table rows
        cleaned = cls.clean_markdown_tables(cleaned)
        return cleaned

