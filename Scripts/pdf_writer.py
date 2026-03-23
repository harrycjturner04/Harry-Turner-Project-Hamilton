#!/usr/bin/env python3
"""Markdown-to-PDF conversion for the biologics analysis pipeline.

Primary backend: WeasyPrint (CSS-styled HTML→PDF with embedded images,
headers/footers, page numbers, and table of contents).

Fallback: reportlab (basic text + image layout).
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, unquote

logger = logging.getLogger("pdf_writer")

# ──────────────────────────────────────────────────────────────────────
# CSS theme for WeasyPrint
# ──────────────────────────────────────────────────────────────────────

REPORT_CSS = """
@page {
    size: A4;
    margin: 2.5cm 2cm 2.5cm 2cm;

    @top-center {
        content: "Biologics Analytical Report";
        font-size: 9pt;
        color: #888;
        font-family: 'DejaVu Sans', 'Helvetica Neue', Arial, sans-serif;
    }

    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9pt;
        color: #888;
        font-family: 'DejaVu Sans', 'Helvetica Neue', Arial, sans-serif;
    }
}

@page :first {
    @top-center { content: none; }
    @bottom-center { content: none; }
}

body {
    font-family: 'DejaVu Sans', 'Helvetica Neue', Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #222;
    max-width: 100%;
}

h1 {
    font-size: 22pt;
    color: #1a1a2e;
    border-bottom: 3px solid #16213e;
    padding-bottom: 8px;
    margin-top: 1.5em;
    page-break-after: avoid;
}

h2 {
    font-size: 16pt;
    color: #16213e;
    border-bottom: 1px solid #ddd;
    padding-bottom: 4px;
    margin-top: 1.2em;
    page-break-after: avoid;
}

h3 {
    font-size: 13pt;
    color: #0f3460;
    margin-top: 1em;
    page-break-after: avoid;
}

p {
    margin: 0.5em 0;
    text-align: justify;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 10pt;
    page-break-inside: avoid;
}

th {
    background-color: #16213e;
    color: white;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
}

td {
    padding: 6px 12px;
    border-bottom: 1px solid #ddd;
}

tr:nth-child(even) {
    background-color: #f8f9fa;
}

code {
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 10pt;
    font-family: 'DejaVu Sans Mono', 'Courier New', monospace;
}

pre {
    background-color: #f4f4f4;
    padding: 12px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 9pt;
    line-height: 1.4;
    page-break-inside: avoid;
}

blockquote {
    border-left: 4px solid #16213e;
    padding-left: 16px;
    color: #555;
    margin: 1em 0;
    font-style: italic;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1em auto;
    page-break-inside: avoid;
}

.figure-caption {
    text-align: center;
    font-size: 10pt;
    color: #555;
    margin-top: 0.3em;
    margin-bottom: 1.5em;
}

ul, ol {
    margin: 0.5em 0;
    padding-left: 2em;
}

li {
    margin: 0.3em 0;
}

.title-page {
    text-align: center;
    padding-top: 8cm;
    page-break-after: always;
}

.title-page h1 {
    font-size: 28pt;
    border: none;
    color: #1a1a2e;
}

.title-page .subtitle {
    font-size: 14pt;
    color: #555;
    margin-top: 1em;
}

.title-page .date {
    font-size: 12pt;
    color: #888;
    margin-top: 2em;
}

.title-page .pipeline-info {
    font-size: 10pt;
    color: #aaa;
    margin-top: 3em;
}

/* Equation styling — renders Unicode-based equations and any $$ blocks */
.equation-block {
    text-align: center;
    font-family: 'DejaVu Sans', 'Helvetica Neue', Arial, sans-serif;
    font-size: 12pt;
    margin: 1em 0;
    padding: 0.5em;
    background-color: #fafafa;
    border-left: 3px solid #16213e;
}

.equation-inline {
    font-family: 'DejaVu Sans', 'Helvetica Neue', Arial, sans-serif;
    font-style: italic;
}
"""

# ──────────────────────────────────────────────────────────────────────
# Markdown → HTML conversion
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# Unicode subscript/superscript → HTML <sub>/<sup>
# ──────────────────────────────────────────────────────────────────────

_UNICODE_SUB_MAP = str.maketrans(
    "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₔₕₖₗₘₙₚₛₜ",
    "0123456789+-=()aeoxəhklmnpst",
)
_UNICODE_SUP_MAP = str.maketrans(
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ",
    "0123456789+-=()ni",
)

_SUB_RE = re.compile(r"[₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₔₕₖₗₘₙₚₛₜ]+")
_SUP_RE = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ]+")


def _unicode_subscripts_to_html(html: str) -> str:
    """Replace Unicode sub/superscript character runs with <sub>/<sup> tags.

    This avoids relying on font coverage for the Unicode subscript block
    (U+2080–U+209F, U+2070–U+207F) which many fonts render as ■.
    """
    def _sub_repl(m: re.Match) -> str:
        return "<sub>" + m.group().translate(_UNICODE_SUB_MAP) + "</sub>"

    def _sup_repl(m: re.Match) -> str:
        return "<sup>" + m.group().translate(_UNICODE_SUP_MAP) + "</sup>"

    html = _SUB_RE.sub(_sub_repl, html)
    html = _SUP_RE.sub(_sup_repl, html)
    return html


# ──────────────────────────────────────────────────────────────────────
# Inline formatting safety net
# ──────────────────────────────────────────────────────────────────────


def _ensure_inline_formatting(html: str) -> str:
    """Convert any surviving **bold** and *italic* markdown to HTML tags.

    Acts as a safety net for cases where the markdown library's conversion
    misses inline formatting (e.g. inside table cells, code-adjacent text,
    or when the fallback converter is used).
    """
    # Bold must be processed before italic to avoid partial matches
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<em>\1</em>', html)
    return html


def _postprocess_equations(html_body: str) -> str:
    """Convert any $$...$$ or $...$ blocks the LLM produced to styled HTML.

    This is a safety net — the primary strategy uses Unicode bold formatting
    in prompts, but if the LLM outputs LaTeX-style notation it should still
    render reasonably.
    """
    # Block equations: $$...$$
    html_body = re.sub(
        r'\$\$(.+?)\$\$',
        r'<div class="equation-block">\1</div>',
        html_body,
        flags=re.DOTALL,
    )
    # Inline equations: $...$  (but not $$)
    html_body = re.sub(
        r'(?<!\$)\$([^$\n]+?)\$(?!\$)',
        r'<span class="equation-inline">\1</span>',
        html_body,
    )
    return html_body


def _markdown_to_html(markdown_text: str, title: str = "Report") -> str:
    """Convert Markdown to styled HTML.

    Uses the `markdown` library if available, otherwise falls back to
    a basic regex-based converter. Post-processes any equation notation.
    """
    try:
        import markdown

        html_body = markdown.markdown(
            markdown_text,
            extensions=["tables", "fenced_code", "toc", "nl2br"],
        )
    except ImportError:
        html_body = _basic_md_to_html(markdown_text)

    # Safety net: convert any $$/$  equation blocks to styled HTML
    html_body = _postprocess_equations(html_body)
    # Convert Unicode sub/superscripts to HTML tags (avoids font coverage gaps)
    html_body = _unicode_subscripts_to_html(html_body)
    # Ensure any surviving **bold**/*italic* markdown is converted
    html_body = _ensure_inline_formatting(html_body)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>{REPORT_CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""


def _basic_md_to_html(text: str) -> str:
    """Minimal Markdown→HTML for when the `markdown` library is unavailable."""
    import html as html_mod

    lines = text.split("\n")
    html_lines = []
    in_code = False
    in_table = False
    in_list = False

    for line in lines:
        stripped = line.strip()

        # Code fences
        if stripped.startswith("```"):
            if in_code:
                html_lines.append("</code></pre>")
                in_code = False
            else:
                html_lines.append("<pre><code>")
                in_code = True
            continue

        if in_code:
            html_lines.append(html_mod.escape(line))
            continue

        # Headings
        if stripped.startswith("######"):
            html_lines.append(f"<h6>{html_mod.escape(stripped[6:].strip())}</h6>")
        elif stripped.startswith("#####"):
            html_lines.append(f"<h5>{html_mod.escape(stripped[5:].strip())}</h5>")
        elif stripped.startswith("####"):
            html_lines.append(f"<h4>{html_mod.escape(stripped[4:].strip())}</h4>")
        elif stripped.startswith("###"):
            html_lines.append(f"<h3>{html_mod.escape(stripped[3:].strip())}</h3>")
        elif stripped.startswith("##"):
            html_lines.append(f"<h2>{html_mod.escape(stripped[2:].strip())}</h2>")
        elif stripped.startswith("#"):
            html_lines.append(f"<h1>{html_mod.escape(stripped[1:].strip())}</h1>")
        # Images
        elif re.match(r"!\[.*?\]\(.*?\)", stripped):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if match:
                alt, src = match.group(1), match.group(2)
                html_lines.append(f'<img src="{html_mod.escape(src)}" alt="{html_mod.escape(alt)}">')
                if alt:
                    html_lines.append(f'<p class="figure-caption">{html_mod.escape(alt)}</p>')
        # Table rows
        elif "|" in stripped and stripped.startswith("|"):
            if not in_table:
                html_lines.append("<table>")
                in_table = True
            # Skip separator rows
            if re.match(r"\|[\s\-:|]+\|", stripped):
                continue
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            tag = "th" if not any("<td>" in l for l in html_lines[-5:] if "<t" in l) else "td"
            row = "".join(f"<{tag}>{html_mod.escape(c)}</{tag}>" for c in cells)
            html_lines.append(f"<tr>{row}</tr>")
        else:
            if in_table:
                html_lines.append("</table>")
                in_table = False
            # List items
            if stripped.startswith("- ") or stripped.startswith("* "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                html_lines.append(f"<li>{html_mod.escape(stripped[2:])}</li>")
            elif re.match(r"^\d+\.\s", stripped):
                if not in_list:
                    html_lines.append("<ol>")
                    in_list = True
                content = re.sub(r"^\d+\.\s", "", stripped)
                html_lines.append(f"<li>{html_mod.escape(content)}</li>")
            else:
                if in_list:
                    html_lines.append("</ul>" if in_list else "</ol>")
                    in_list = False
                if stripped:
                    # Bold and italic
                    processed = html_mod.escape(stripped)
                    processed = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", processed)
                    processed = re.sub(r"\*(.+?)\*", r"<em>\1</em>", processed)
                    html_lines.append(f"<p>{processed}</p>")

    if in_table:
        html_lines.append("</table>")
    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)


# ──────────────────────────────────────────────────────────────────────
# Title page generation
# ──────────────────────────────────────────────────────────────────────


def _build_title_page_html(title: str) -> str:
    """Generate HTML for a styled title page."""
    date_str = datetime.now().strftime("%d %B %Y")
    return f"""
<div class="title-page">
    <h1>{title}</h1>
    <p class="subtitle">Biologics Analytical Report</p>
    <p class="date">{date_str}</p>
    <p class="pipeline-info">Generated by the Biologics Analysis Pipeline</p>
</div>
"""


# ──────────────────────────────────────────────────────────────────────
# Image path resolution
# ──────────────────────────────────────────────────────────────────────


def _resolve_image_paths(html: str, base_dir: Optional[Path] = None) -> str:
    """Convert relative image paths to absolute file:// URIs for WeasyPrint.

    Handles URL-encoded paths (spaces → %20) so WeasyPrint can resolve them.
    """
    if base_dir is None:
        return html

    def _resolve(match):
        src = match.group(1)
        if src.startswith(("http://", "https://", "file://", "data:")):
            return match.group(0)
        # Decode any existing URL-encoding before resolving
        decoded_src = unquote(src)
        src_path = Path(decoded_src)
        if src_path.is_absolute():
            abs_path = src_path
        else:
            abs_path = (base_dir / decoded_src).resolve()
        if abs_path.exists():
            # URL-encode the path (preserve / and :) for valid file:// URI
            encoded = quote(str(abs_path), safe="/:")
            return f'src="file://{encoded}"'
        return match.group(0)

    return re.sub(r'src="([^"]+)"', _resolve, html)


# ──────────────────────────────────────────────────────────────────────
# Report HTML validation
# ──────────────────────────────────────────────────────────────────────


def validate_report_html(html: str) -> List[Dict[str, str]]:
    """Check rendered HTML for common rendering failures.

    Returns a list of warning dicts. An empty list means no issues found.
    This is a safety net — the preprocessing functions should have already
    handled these, but this catches regressions.
    """
    warnings: List[Dict[str, str]] = []
    # 1. Surviving markdown image syntax (not converted to <img>)
    surviving_imgs = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', html)
    if surviving_imgs:
        warnings.append({
            "type": "unrendered_image",
            "detail": f"Markdown image syntax survived conversion ({len(surviving_imgs)} instance(s))",
        })
    # 2. Surviving bold/italic asterisks (outside of <code>/<pre> blocks)
    # Strip code blocks before checking to avoid false positives
    no_code = re.sub(r'<(code|pre)>.*?</\1>', '', html, flags=re.DOTALL)
    if re.search(r'\*\*[^<]+\*\*', no_code):
        warnings.append({
            "type": "unrendered_bold",
            "detail": "Bold markdown (**text**) survived conversion",
        })
    # 3. Unicode subscript/superscript characters still present
    if _SUB_RE.search(html) or _SUP_RE.search(html):
        warnings.append({
            "type": "unicode_subscript",
            "detail": "Unicode sub/superscripts not converted to HTML tags",
        })
    # 4. Broken image references (src points to nonexistent file)
    for m in re.finditer(r'<img[^>]+src="([^"]+)"', html):
        src = m.group(1)
        if src.startswith("file://"):
            file_path = unquote(src[7:])  # strip file:// prefix
            if not os.path.exists(file_path):
                warnings.append({
                    "type": "missing_image",
                    "detail": f"Image not found: {file_path}",
                })
    return warnings


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def markdown_to_pdf(
    markdown_text: str,
    output_path: str | Path,
    title: str = "Analysis Report",
    plot_paths: Optional[List[str]] = None,
    include_title_page: bool = True,
) -> Path:
    """Convert a Markdown report to a styled PDF.

    Args:
        markdown_text: The report in Markdown format.
        output_path: Destination PDF path.
        title: Report title (used for title page and header).
        plot_paths: Optional list of plot image paths to append if not
            already referenced in the markdown.
        include_title_page: Whether to prepend a styled title page.

    Returns:
        The Path to the generated PDF.

    Raises:
        RuntimeError: If neither WeasyPrint nor reportlab is available.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Append unreferenced plots
    if plot_paths:
        for pp in plot_paths:
            if pp not in markdown_text:
                fig_name = Path(pp).stem.replace("_", " ").title()
                encoded_pp = quote(pp, safe="/:")
                markdown_text += f"\n\n![{fig_name}]({encoded_pp})\n"

    # Build HTML
    title_html = _build_title_page_html(title) if include_title_page else ""
    body_html = _markdown_to_html(markdown_text, title=title)
    # Inject title page after <body>
    if title_html:
        body_html = body_html.replace("<body>", f"<body>\n{title_html}", 1)

    # Resolve image paths relative to output directory
    body_html = _resolve_image_paths(body_html, base_dir=output_path.parent)

    # Validate HTML before rendering — log warnings but don't block
    html_warnings = validate_report_html(body_html)
    if html_warnings:
        logger.warning(
            "Report HTML validation: %d warning(s): %s",
            len(html_warnings), html_warnings,
        )

    # Try WeasyPrint first
    try:
        from weasyprint import HTML

        HTML(string=body_html).write_pdf(str(output_path))
        logger.info("PDF generated with WeasyPrint: %s", output_path)
        return output_path
    except ImportError:
        logger.info("WeasyPrint not available, trying reportlab fallback")
    except Exception as exc:
        logger.warning("WeasyPrint failed (%s), trying reportlab fallback", exc)

    # Fallback: reportlab
    try:
        return _reportlab_fallback(markdown_text, output_path, title, plot_paths)
    except ImportError:
        raise RuntimeError(
            "Neither WeasyPrint nor reportlab is installed. "
            "Install one with: pip install weasyprint  OR  pip install reportlab"
        )


def _register_dejavu_fonts() -> str:
    """Register DejaVu Sans font family with reportlab and return the face name.

    Returns 'DejaVu Sans' on success, 'Helvetica' if fonts are not found.
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    _DEJAVU_DIR = Path("/usr/share/fonts/dejavu")
    _VARIANTS = {
        "DejaVu Sans": "DejaVuSans.ttf",
        "DejaVu Sans Bold": "DejaVuSans-Bold.ttf",
        "DejaVu Sans Italic": "DejaVuSans-Oblique.ttf",
        "DejaVu Sans BoldItalic": "DejaVuSans-BoldOblique.ttf",
    }

    try:
        for face_name, filename in _VARIANTS.items():
            font_path = _DEJAVU_DIR / filename
            if font_path.exists():
                pdfmetrics.registerFont(TTFont(face_name, str(font_path)))
            else:
                logger.debug("DejaVu font not found: %s", font_path)
                return "Helvetica"

        # Register the family so <b>/<i> tags map correctly
        from reportlab.lib.fonts import addMapping
        addMapping("DejaVu Sans", 0, 0, "DejaVu Sans")
        addMapping("DejaVu Sans", 1, 0, "DejaVu Sans Bold")
        addMapping("DejaVu Sans", 0, 1, "DejaVu Sans Italic")
        addMapping("DejaVu Sans", 1, 1, "DejaVu Sans BoldItalic")
        logger.debug("Registered DejaVu Sans font family with reportlab")
        return "DejaVu Sans"
    except Exception as exc:
        logger.warning("Failed to register DejaVu fonts: %s", exc)
        return "Helvetica"


def _md_line_to_paragraph_xml(line: str) -> str:
    """Convert a single markdown line to reportlab Paragraph XML.

    Handles: bold, italic, inline code, Unicode sub/superscripts.
    Must be called AFTER XML-escaping &, <, >.
    """
    # XML-escape first
    s = (
        line.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    # Bold: **text** → <b>text</b>
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    # Italic: *text* → <i>text</i> (not inside bold)
    s = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", s)
    # Inline code: `text` → <font face="Courier">text</font>
    s = re.sub(r"`([^`]+?)`", r'<font face="Courier">\1</font>', s)
    # Unicode subscripts → <sub> (reportlab Paragraph supports this)
    s = _SUB_RE.sub(lambda m: "<sub>" + m.group().translate(_UNICODE_SUB_MAP) + "</sub>", s)
    # Unicode superscripts → <sup>
    s = _SUP_RE.sub(lambda m: "<sup>" + m.group().translate(_UNICODE_SUP_MAP) + "</sup>", s)
    return s


def _reportlab_fallback(
    markdown_text: str,
    output_path: Path,
    title: str,
    plot_paths: Optional[List[str]],
) -> Path:
    """Production-quality PDF generation using reportlab.

    Registers DejaVu Sans for Greek/mathematical symbol support, converts
    Unicode sub/superscripts to <sub>/<sup> tags, and renders headings,
    lists, images, and body text via reportlab Paragraph.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
        ListFlowable, ListItem,
    )
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.colors import HexColor

    font_name = _register_dejavu_fonts()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Override default styles with DejaVu Sans
    for style_name in ("Normal", "Heading1", "Heading2", "Heading3",
                        "Heading4", "Title", "BodyText"):
        if style_name in styles:
            styles[style_name].fontName = font_name

    styles.add(ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=24,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        "FigureCaption",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=10,
        alignment=TA_CENTER,
        textColor=HexColor("#555555"),
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        "ListBody",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=11,
        leading=16,
        leftIndent=18,
        bulletIndent=6,
    ))

    # Style headings with pipeline brand colours
    styles["Heading1"].textColor = HexColor("#1a1a2e")
    styles["Heading1"].fontSize = 18
    styles["Heading1"].spaceBefore = 16
    styles["Heading1"].spaceAfter = 8
    styles["Heading2"].textColor = HexColor("#16213e")
    styles["Heading2"].fontSize = 15
    styles["Heading2"].spaceBefore = 12
    styles["Heading2"].spaceAfter = 6
    styles["Heading3"].textColor = HexColor("#0f3460")
    styles["Heading3"].fontSize = 13
    styles["Heading3"].spaceBefore = 10
    styles["Heading3"].spaceAfter = 4

    story = []

    # Title page
    story.append(Spacer(1, 8 * cm))
    story.append(Paragraph(title, styles["ReportTitle"]))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("Biologics Analytical Report", styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(datetime.now().strftime("%d %B %Y"), styles["Normal"]))
    story.append(PageBreak())

    # Track images already embedded inline so we skip them in the appendix
    _embedded_images: set = set()

    # Body text — Paragraph-based rendering with inline formatting
    for line in markdown_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 6))
            continue

        # Headings (check longest prefix first)
        if stripped.startswith("#"):
            for level, prefix in ((4, "#### "), (3, "### "), (2, "## "), (1, "# ")):
                if stripped.startswith(prefix):
                    heading_text = _md_line_to_paragraph_xml(stripped[len(prefix):])
                    style_key = f"Heading{min(level, 3)}"
                    story.append(Paragraph(heading_text, styles[style_key]))
                    break
            else:
                # More than 4 hashes — treat as Heading3
                text = stripped.lstrip("#").strip()
                story.append(Paragraph(_md_line_to_paragraph_xml(text), styles["Heading3"]))
            continue

        # Inline images: ![alt](path)
        img_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if img_match:
            alt_text, img_src = img_match.group(1), img_match.group(2)
            img_path = Path(unquote(img_src))
            if img_path.exists() and img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                try:
                    img = Image(str(img_path), width=16 * cm, height=10 * cm)
                    img.hAlign = "CENTER"
                    story.append(Spacer(1, 8))
                    story.append(img)
                    if alt_text:
                        story.append(Paragraph(
                            _md_line_to_paragraph_xml(alt_text),
                            styles["FigureCaption"],
                        ))
                    _embedded_images.add(str(img_path))
                except Exception as exc:
                    logger.warning("Could not embed inline image %s: %s", img_path, exc)
            continue

        # Unordered list items: - or *
        if stripped.startswith("- ") or stripped.startswith("* "):
            item_text = _md_line_to_paragraph_xml(stripped[2:])
            story.append(Paragraph(
                f"\u2022 {item_text}",
                styles["ListBody"],
            ))
            continue

        # Ordered list items: 1. 2. etc.
        ol_match = re.match(r"^(\d+)\.\s+(.+)", stripped)
        if ol_match:
            num, content = ol_match.group(1), ol_match.group(2)
            item_text = _md_line_to_paragraph_xml(content)
            story.append(Paragraph(
                f"{num}. {item_text}",
                styles["ListBody"],
            ))
            continue

        # Horizontal rule
        if re.match(r"^[-*_]{3,}\s*$", stripped):
            story.append(Spacer(1, 12))
            continue

        # Regular paragraph
        story.append(Paragraph(
            _md_line_to_paragraph_xml(stripped), styles["ReportBody"],
        ))

    # Append unreferenced plot images
    if plot_paths:
        unreferenced = [
            pp for pp in plot_paths
            if pp not in _embedded_images and str(pp) not in _embedded_images
        ]
        if unreferenced:
            story.append(PageBreak())
            story.append(Paragraph("Figures", styles["Heading1"]))
            for pp in unreferenced:
                p = Path(pp)
                if p.exists() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
                    try:
                        img = Image(str(p), width=16 * cm, height=10 * cm)
                        img.hAlign = "CENTER"
                        story.append(Spacer(1, 12))
                        story.append(img)
                        story.append(Paragraph(
                            p.stem.replace("_", " ").title(),
                            styles["FigureCaption"],
                        ))
                    except Exception as exc:
                        logger.warning("Could not embed image %s: %s", p, exc)

    doc.build(story)
    logger.info("PDF generated with reportlab (font=%s): %s", font_name, output_path)
    return output_path
