
import re
import io

# Recognises the content-section heading in any form
CONTENT_HEADING_RE = re.compile(
    r"(course\s*(outline|content|description|topics?|syllabus|plan|overview"
    r"|modules?|units?|curriculum)"
    r"|weekly\s*(distribution|schedule|plan|outline|topics?|breakdown)"
    r"|lecture\s*(plan|schedule|outline|topics?)"
    r"|topics?\s*(covered|to\s*be\s*covered)"
    r"|course\s*schedule"
    r"|syllabus)",
    re.IGNORECASE,
)

# Column header that means "this column holds the course content"
CONTENT_COLUMN_RE = re.compile(
    r"(topic|content|course\s*content|weekly\s*(topic|content)"
    r"|lecture|description|coverage|outline)",
    re.IGNORECASE,
)


def _is_content_heading(text: str) -> bool:
    return bool(CONTENT_HEADING_RE.search(text.strip()))


def _is_content_column(header: str) -> bool:
    return bool(CONTENT_COLUMN_RE.search(header.strip()))


# ═══════════════════════════════════════════════════════════════════════════════
#  DOCX
# ═══════════════════════════════════════════════════════════════════════════════

def extract_from_docx(file_bytes: bytes) -> tuple[str, str]:
    import docx
    doc = docx.Document(io.BytesIO(file_bytes))

    # Pass 1 — non-table paragraphs (preferred)
    result = _docx_non_table(doc)
    if result:
        return result, "non-table"

    # Pass 2 — table fallback
    result = _docx_table(doc)
    if result:
        return result, "table"

    return "", "not found"


def _docx_non_table(doc) -> str:
    collecting = False
    lines: list[str] = []

    for para in doc.paragraphs:
        text       = para.text.strip()
        is_heading = para.style.name.lower().startswith("heading")
        is_bold    = _is_bold_para(para)

        if not text:
            continue

        # Found the content heading (any format) → start collecting
        if _is_content_heading(text):
            collecting = True
            continue

        # While collecting: stop on the next heading (styled or bold)
        if collecting and (is_heading or is_bold):
            break

        if collecting:
            lines.append(text)

    return "\n".join(lines)


def _is_bold_para(para) -> bool:
    """
    True if the paragraph is short (< 80 chars) and every non-empty
    run is bold — used as a manual/informal heading.
    """
    text = para.text.strip()
    if not text or len(text) > 80:
        return False
    runs = [r for r in para.runs if r.text.strip()]
    return bool(runs) and all(r.bold for r in runs)


def _docx_table(doc) -> str:
    """
    Walk every table in the document.
    For each table, look at the header row to find a column whose header
    matches CONTENT_COLUMN_RE.  Read only that column's cells.
    Stop when the table ends (natural boundary).
    """
    for table in doc.tables:
        rows = table.rows
        if len(rows) < 2:
            continue

        # Find which column index is the content column
        header_cells = [cell.text.strip() for cell in rows[0].cells]
        col_idx = None
        for idx, header in enumerate(header_cells):
            if _is_content_column(header):
                col_idx = idx
                break

        if col_idx is None:
            continue

        # Read only the content column, skip the header row
        lines: list[str] = []
        seen:  set[str]  = set()
        for row in rows[1:]:
            if col_idx >= len(row.cells):
                continue
            text = row.cells[col_idx].text.strip()
            if text and text not in seen:   # seen handles merged/repeated cells
                seen.add(text)
                lines.append(text)

        if lines:
            return "\n".join(lines)

    return ""


# ═══════════════════════════════════════════════════════════════════════════════
#  PDF
# ═══════════════════════════════════════════════════════════════════════════════

def extract_from_pdf(file_bytes: bytes) -> tuple[str, str]:
    import pdfplumber

    all_lines:  list[str]       = []
    all_tables: list[list[list]]= []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                all_lines.extend(txt.splitlines())
            tbls = page.extract_tables()
            if tbls:
                all_tables.extend(tbls)

    # OCR fallback for scanned / image PDFs
    if len(" ".join(all_lines).strip()) < 200:
        all_lines = _ocr_pdf(file_bytes)
        result = _pdf_non_table(all_lines)
        return (result, "tesseract-ocr") if result else ("\n".join(all_lines[:100]), "tesseract-ocr (raw)")

    # Pass 1 — non-table (preferred), but only if it found real topic content
    # (at least 5 lines — avoids matching brief info-table rows like Semester/Section)
    result = _pdf_non_table(all_lines)
    if result and len(result.splitlines()) >= 5:
        return result, "non-table"

    # Pass 2 — table fallback
    result = _pdf_table(all_tables)
    if result:
        return result, "table"

    # Last resort: return whatever non-table found even if short
    if result:
        return result, "non-table (short)"

    return "", "not found"


def _pdf_non_table(lines: list[str]) -> str:
    """
    Find the line matching CONTENT_HEADING_RE.
    Collect every line after it.

    After the content heading there may be sub-header lines like "Week" or
    "(Assignment & Quizzes)" that appear before real content starts.
    We skip those until we hit the first real content line (has a digit OR
    is long enough to be a topic sentence).

    Stop when a new short heading-like line appears AFTER real content
    has already been collected.
    """
    collecting = False
    warmed_up  = False   # True once first real content line is seen
    result: list[str] = []

    for line in lines:
        text = line.strip()
        if not text:
            continue

        # Found the content heading → start collecting
        if _is_content_heading(text):
            collecting = True
            continue

        if collecting:
            has_digit = bool(re.search(r"\d", text))
            is_short  = len(text) < 60

            # Skip sub-header lines before real content starts
            if not warmed_up and is_short and not has_digit:
                continue

            warmed_up = True   # real content line reached

            # Stop on a new short heading-like line after content has started
            if is_short and not has_digit and not _is_content_heading(text):
                break

            # Strip PDF bullet unicode chars (e.g. \uf0b7)
            text = re.sub(r"[\uf0b0-\uf0ff]", "", text).strip()
            if text:
                result.append(text)

    return "\n".join(result)


def _pdf_table(tables: list[list[list]]) -> str:
    """
    Walk pdfplumber tables.
    Find the one that has a column matching CONTENT_COLUMN_RE.
    Read only that column.  Stop when the table ends.

    Handles the case where the header row and data rows are split across
    different pdfplumber table objects (common when a table spans pages).
    We remember the last matched col_idx and apply it to the next table
    if that table has no header row of its own.
    """
    last_col_idx: int | None = None   # carry forward across page-split tables

    result_lines: list[str] = []
    seen: set[str] = set()

    for table in tables:
        if not table:
            continue

        header = [str(c).strip() if c else "" for c in table[0]]
        col_idx = None

        # Check if this table has a recognisable content column header
        for idx, h in enumerate(header):
            if _is_content_column(h):
                col_idx = idx
                last_col_idx = idx
                break

        # If no header found but the previous table set a col_idx, this table
        # is likely the continuation (data-only, page 2+)
        if col_idx is None and last_col_idx is not None:
            col_idx = last_col_idx
            data_rows = table          # no header row to skip
        else:
            data_rows = table[1:]      # skip the header row

        if col_idx is None:
            last_col_idx = None        # reset — unrelated table
            continue

        for row in data_rows:
            if col_idx >= len(row):
                continue
            text = str(row[col_idx]).strip() if row[col_idx] else ""
            # Strip PDF bullet chars
            text = re.sub(r"[\uf0b0-\uf0ff]", "", text).strip()
            if text and text not in seen:
                seen.add(text)
                result_lines.append(text)

    return "\n".join(result_lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  OCR
# ═══════════════════════════════════════════════════════════════════════════════

def _ocr_pdf(file_bytes: bytes) -> list[str]:
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(file_bytes, dpi=200)
        lines: list[str] = []
        for img in images:
            lines.extend(pytesseract.image_to_string(img).splitlines())
        return lines
    except Exception as e:
        raise RuntimeError(f"OCR failed: {e}") from e