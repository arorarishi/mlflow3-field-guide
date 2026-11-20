"""Build book/Offline-to-Online.pdf from the chapter READMEs and example code.

    ./book.sh pdf            (or: .venv/bin/python book/build_pdf.py)

Handles the Markdown subset the book uses: headings, paragraphs, lists, block quotes,
fenced code and pipe tables.
"""
from __future__ import annotations

import datetime
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

BOOK = Path(__file__).resolve().parent
OUT = BOOK / "Offline-to-Online.pdf"
PARTS = {
    1: "Part I — Tracking",
    7: "Part II — Models",
    13: "Part III — Workflow",
    16: "Part IV — GenAI",
    23: "Part V — Operations",
}

# ---------------------------------------------------------------- fonts
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
if (FONT_DIR / "DejaVuSans.ttf").exists():
    for name, file in {
        "Sans": "DejaVuSans.ttf",
        "Sans-Bold": "DejaVuSans-Bold.ttf",
        "Sans-Oblique": "DejaVuSans-Oblique.ttf",
        "Sans-BoldOblique": "DejaVuSans-BoldOblique.ttf",
        "Mono": "DejaVuSansMono.ttf",
    }.items():
        if (FONT_DIR / file).exists():
            pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
    registered = set(pdfmetrics.getRegisteredFontNames())
    italic = "Sans-Oblique" if "Sans-Oblique" in registered else "Sans"  # some distros ship no oblique
    bold_italic = "Sans-BoldOblique" if "Sans-BoldOblique" in registered else "Sans-Bold"
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic=italic, boldItalic=bold_italic)
    SANS, BOLD, MONO = "Sans", "Sans-Bold", "Mono"
else:  # built-in fonts lack some symbols (✓, ▶, box drawing), but the PDF still builds
    SANS, BOLD, MONO = "Helvetica", "Helvetica-Bold", "Courier"

ACCENT = colors.HexColor("#0B6E99")
CODE_BG = colors.HexColor("#F4F6F8")
RULE = colors.HexColor("#D0D7DE")

base = getSampleStyleSheet()
S = {
    "body": ParagraphStyle("body", base["Normal"], fontName=SANS, fontSize=9.5, leading=13.5, spaceAfter=5),
    "h1": ParagraphStyle("h1", fontName=BOLD, fontSize=20, leading=25, textColor=ACCENT, spaceAfter=10),
    "h2": ParagraphStyle("h2", fontName=BOLD, fontSize=13, leading=17, spaceBefore=10, spaceAfter=5),
    "h3": ParagraphStyle("h3", fontName=BOLD, fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=4),
    "part": ParagraphStyle("part", fontName=BOLD, fontSize=26, leading=32, textColor=ACCENT, alignment=TA_CENTER),
    "quote": ParagraphStyle("quote", fontName=SANS, fontSize=9.5, leading=13.5, leftIndent=10,
                            textColor=colors.HexColor("#444444"), borderColor=ACCENT, borderWidth=0,
                            borderPadding=(2, 0, 2, 6), spaceAfter=6),
    "code": ParagraphStyle("code", fontName=MONO, fontSize=7.4, leading=9.4, backColor=CODE_BG,
                           borderPadding=5, spaceBefore=3, spaceAfter=9, leftIndent=5, rightIndent=5),
    "cell": ParagraphStyle("cell", fontName=SANS, fontSize=8.3, leading=10.8),
    "cellh": ParagraphStyle("cellh", fontName=BOLD, fontSize=8.3, leading=10.8),
    "title": ParagraphStyle("title", fontName=BOLD, fontSize=34, leading=40, alignment=TA_CENTER, textColor=ACCENT),
    "subtitle": ParagraphStyle("subtitle", fontName=SANS, fontSize=13, leading=18, alignment=TA_CENTER),
    "toc": ParagraphStyle("toc", fontName=SANS, fontSize=10, leading=15, leftIndent=12),
    "tocpart": ParagraphStyle("tocpart", fontName=BOLD, fontSize=11, leading=16, spaceBefore=8),
}


# ---------------------------------------------------------------- markdown -> flowables
def inline(text: str) -> str:
    """Escape XML and convert inline Markdown (code, bold, italic, links)."""
    codes: list[str] = []

    def keep_code(match):
        codes.append(match.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", keep_code, text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    for i, code in enumerate(codes):
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00{i}\x00", f'<font name="{MONO}" size="8.3" color="#8A1F3C">{escaped}</font>')
    return text


def code_block(source: str) -> Preformatted:
    return Preformatted(source.rstrip("\n"), S["code"], maxLineLength=104, newLineChars="  ")


def table(rows: list[list[str]], width: float) -> Table:
    header, *body = rows
    data = [[Paragraph(inline(c), S["cellh"]) for c in header]]
    data += [[Paragraph(inline(c), S["cell"]) for c in row + [""] * (len(header) - len(row))] for row in body]
    t = Table(data, colWidths=[width / len(header)] * len(header), repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F1F6")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def markdown(md: str, width: float, chapter_heading: bool = True) -> list:
    out: list = []
    para: list[str] = []
    bullets: list[str] = []
    ordered = False
    lines = md.splitlines()
    i = 0

    def flush():
        nonlocal para, bullets
        if para:
            out.append(Paragraph(inline(" ".join(para)), S["body"]))
            para = []
        if bullets:
            items = [ListItem(Paragraph(inline(b), S["body"]), leftIndent=12) for b in bullets]
            out.append(ListFlowable(items, bulletType="1" if ordered else "bullet", start="1" if ordered else "•",
                                    leftIndent=12, bulletFontName=SANS, bulletFontSize=8.5))
            bullets = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            out.append(code_block("\n".join(block)))
        elif stripped.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            out.append(table(rows, width))
            out.append(Spacer(1, 6))
            continue
        elif m := re.match(r"^(#{1,3})\s+(.*)", line):
            flush()
            level = len(m.group(1))
            style = S["h1"] if level == 1 and chapter_heading else S["h2"] if level <= 2 else S["h3"]
            out.append(Paragraph(inline(m.group(2)), style))
        elif m := re.match(r"^\s*([-*]|\d+\.)\s+(.*)", line):
            if para:
                flush()
            if bullets and ordered != m.group(1)[0].isdigit():
                flush()
            ordered = m.group(1)[0].isdigit()
            bullets.append(m.group(2))
        elif stripped.startswith(">"):
            flush()
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append(Paragraph(inline(" ".join(quote)), S["quote"]))
            continue
        elif not stripped:
            flush()
        elif bullets and line.startswith("  "):
            bullets[-1] += " " + stripped  # continuation of a list item
        else:
            para.append(stripped)
        i += 1
    flush()
    return out


# ---------------------------------------------------------------- document
class BookDoc(BaseDocTemplate):
    def __init__(self, path: str):
        super().__init__(path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                         topMargin=18 * mm, bottomMargin=18 * mm,
                         title="Offline to Online: Evaluating and Shipping LLM Apps with MLflow 3", author="MLflow learning repo", subject="MLflow 3.18")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([
            PageTemplate("cover", [frame]),
            PageTemplate("page", [frame], onPage=self.footer),
        ])

    @staticmethod
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(SANS, 7.5)
        canvas.setFillColor(colors.grey)
        canvas.drawString(doc.leftMargin, 10 * mm, "Offline to Online")
        canvas.drawRightString(A4[0] - doc.rightMargin, 10 * mm, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        """PDF bookmarks: parts at level 0, chapters at level 1."""
        if isinstance(flowable, Paragraph) and flowable.style.name in ("part", "h1"):
            text = flowable.getPlainText()
            key = f"k{id(flowable)}"
            self.canv.bookmarkPage(key)
            is_part = flowable.style.name == "part"
            if is_part:
                self._in_part = True
            # Front matter and appendix headings sit at the top level.
            in_part = getattr(self, "_in_part", False) and not text.startswith("Appendix")
            self.canv.addOutlineEntry(text, key, level=0 if is_part or not in_part else 1)


def chapters() -> list[Path]:
    return sorted(p for p in BOOK.iterdir() if p.is_dir() and re.match(r"\d\d-", p.name))


def build() -> Path:
    doc = BookDoc(str(OUT))
    width = doc.width
    story: list = []

    # Cover
    story += [Spacer(1, 70 * mm), Paragraph("Offline to Online", S["title"]), Spacer(1, 4 * mm),
              Paragraph("Evaluating and Shipping LLM Apps with MLflow 3", S["subtitle"]), Spacer(1, 8 * mm),
              Paragraph("25 hands-on chapters: offline evals, online tracing and feedback, prompts, agents,<br/>"
                        "plus the tracking, registry, serving and ops that get them to production", S["subtitle"]),
              Spacer(1, 10 * mm),
              Paragraph(f"Written for MLflow 3.18.0 · built {datetime.date.today():%d %B %Y}", S["subtitle"])]
    story.append(PageBreak())

    # Contents
    from reportlab.platypus.doctemplate import NextPageTemplate

    story.insert(-1, NextPageTemplate("page"))
    story.append(Paragraph("Contents", S["h1"]))
    for chapter in chapters():
        n = int(chapter.name[:2])
        if n in PARTS:
            story.append(Paragraph(PARTS[n], S["tocpart"]))
        title = (chapter / "README.md").read_text().splitlines()[0].lstrip("# ")
        match = re.search(r"\*\*You will learn:\*\*(.*?)(?:\n\n|$)", (chapter / "README.md").read_text(), re.S)
        learn = " ".join(match.group(1).split()) if match else ""
        story.append(Paragraph(f"<b>{inline(title)}</b><br/><font size='8.5' color='#555555'>"
                               f"{inline(learn.replace('**You will learn:**', '').strip())}</font>", S["toc"]))
    story.append(PageBreak())

    # Introduction: the repository README, minus its contents tables (already above)
    intro = (BOOK.parent / "README.md").read_text()
    intro = re.sub(r"## Contents.*?(?=## Suggested paths)", "", intro, flags=re.S)
    intro = re.sub(r"\]\((?:\.\./)?book/", "](", intro)
    story += markdown(intro, width)

    # Chapters, each followed by its example code
    for chapter in chapters():
        n = int(chapter.name[:2])
        if n in PARTS:
            story += [PageBreak(), Spacer(1, 90 * mm), Paragraph(PARTS[n], S["part"])]
        story.append(PageBreak())
        story += markdown((chapter / "README.md").read_text(), width)
        sources = [chapter / "example.py"] + sorted(
            p for p in chapter.rglob("*") if p.is_file() and p.name != "README.md" and p.name != "example.py"
            and "__pycache__" not in p.parts
        )
        for source in sources:
            rel = source.relative_to(BOOK)
            story.append(KeepTogether([Paragraph(f"Code: book/{rel}", S["h3"])]))
            story.append(code_block(source.read_text()))

    # Appendix: shared helpers
    story += [PageBreak(), Paragraph("Appendix — shared code", S["h1"])]
    for name in ("shared.py", "support_bot.py"):
        story += [Paragraph(f"book/{name}", S["h3"]), code_block((BOOK / name).read_text())]

    doc.build(story)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")
