"""Build the PDF versions of the project report and the learning guide.

    python scripts/build_pdfs.py

Reads docs/report.md and docs/learning_guide.md (the single sources of truth) and
writes docs/pdf/MOSAIC_Project_Report.pdf and docs/pdf/MOSAIC_Learning_Guide.pdf.
Formulas are rendered with matplotlib's mathtext, diagrams are drawn with
matplotlib (Mermaid cannot be rendered in a PDF), figures come from the real
experiment run in reports/experiments/main/figures.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm, mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, Preformatted, Spacer, Table, TableStyle,
)
from reportlab.platypus.doctemplate import BaseDocTemplate, Frame, PageTemplate  # noqa: E402
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "pdf"
CACHE = OUT / ".render_cache"

BRAND = colors.HexColor("#134846")
BRAND_LIGHT = colors.HexColor("#eef8f7")
ACCENT = colors.HexColor("#2a8c80")
INK = colors.HexColor("#1f2933")
SOFT = colors.HexColor("#52606d")
RULE = colors.HexColor("#d9e2ec")
NOTE_BG = colors.HexColor("#fff8e6")
NOTE_BORDER = colors.HexColor("#f0b429")

# ----------------------------------------------------------------- fonts
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-BoldOblique", str(FONT_DIR / "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuMono", str(FONT_DIR / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique",
                              boldItalic="DejaVu-BoldOblique")

BASE = ParagraphStyle("base", fontName="DejaVu", fontSize=9.6, leading=13.6, textColor=INK, spaceAfter=5)
STYLES = {
    "body": BASE,
    "h1": ParagraphStyle("h1", parent=BASE, fontName="DejaVu-Bold", fontSize=18, leading=23, textColor=BRAND,
                         spaceBefore=4, spaceAfter=10),
    "h2": ParagraphStyle("h2", parent=BASE, fontName="DejaVu-Bold", fontSize=13.5, leading=18, textColor=BRAND,
                         spaceBefore=12, spaceAfter=6),
    "h3": ParagraphStyle("h3", parent=BASE, fontName="DejaVu-Bold", fontSize=11, leading=15, textColor=INK,
                         spaceBefore=9, spaceAfter=4),
    "h4": ParagraphStyle("h4", parent=BASE, fontName="DejaVu-Bold", fontSize=10, leading=14, spaceBefore=6),
    "bullet": ParagraphStyle("bullet", parent=BASE, leftIndent=14, bulletIndent=4, spaceAfter=2.5),
    "bullet2": ParagraphStyle("bullet2", parent=BASE, leftIndent=28, bulletIndent=18, spaceAfter=2),
    "quote": ParagraphStyle("quote", parent=BASE, textColor=INK, leftIndent=8, rightIndent=4, spaceAfter=0),
    "cell": ParagraphStyle("cell", parent=BASE, fontSize=8.2, leading=10.6, spaceAfter=0),
    "cell_small": ParagraphStyle("cell_small", parent=BASE, fontSize=7, leading=9, spaceAfter=0),
    "cellh": ParagraphStyle("cellh", parent=BASE, fontName="DejaVu-Bold", fontSize=8.2, leading=10.6,
                            textColor=colors.white, spaceAfter=0),
    "cellh_small": ParagraphStyle("cellh_small", parent=BASE, fontName="DejaVu-Bold", fontSize=7, leading=9,
                                  textColor=colors.white, spaceAfter=0),
    "code": ParagraphStyle("code", fontName="DejaVuMono", fontSize=8, leading=10.5, textColor=INK),
    "caption": ParagraphStyle("caption", parent=BASE, fontSize=8.2, leading=11, textColor=SOFT, alignment=TA_CENTER,
                              spaceAfter=10),
    "toc0": ParagraphStyle("toc0", parent=BASE, fontName="DejaVu-Bold", fontSize=10, leading=15, leftIndent=0),
    "toc1": ParagraphStyle("toc1", parent=BASE, fontSize=9, leading=12.5, leftIndent=16, textColor=SOFT),
}


# ------------------------------------------------------------------ math
def _prep_tex(tex: str) -> str:
    tex = tex.strip()
    tex = re.sub(r"\\operatorname\{([^}]*)\}", r"\\mathrm{\1}", tex)
    tex = re.sub(r"\\text\{([^}]*)\}", lambda m: r"\mathrm{" + m.group(1).replace(" ", r"\ ") + "}", tex)
    tex = re.sub(r"\\tfrac(\d)(\w)", r"\\frac{\1}{\2}", tex)
    tex = tex.replace(r"\tfrac", r"\frac").replace(r"\lvert", "|").replace(r"\rvert", "|")
    tex = tex.replace(r"\|", r"\Vert ").replace(r"\qquad", r"\quad\quad")
    tex = re.sub(r"\\ge(?![a-z])", r"\\geq", tex)
    tex = re.sub(r"\\mathbf\s+(\w)", r"\\mathbf{\1}", tex)
    tex = re.sub(r"\\hat\s*\\sigma", r"\\hat{\\sigma}", tex)
    tex = re.sub(r"\\tilde\s+(\w)", r"\\tilde{\1}", tex)
    return tex


def math_image(tex: str, fontsize: float) -> tuple[Path, float, float] | None:
    """Render TeX with mathtext to a PNG; returns (path, width_pt, height_pt) or None."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{tex}|{fontsize}".encode()).hexdigest()[:16]
    path = CACHE / f"m_{key}.png"
    meta = CACHE / f"m_{key}.txt"
    if path.exists() and meta.exists():
        w, h = map(float, meta.read_text().split())
        return path, w, h
    fig = plt.figure(figsize=(0.01, 0.01))
    try:
        txt = fig.text(0, 0, f"${_prep_tex(tex)}$", fontsize=fontsize, color="#1f2933")
        fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.02)
    except Exception as exc:  # unsupported construct -> caller falls back to text
        print(f"  math fallback ({exc.__class__.__name__}): {tex[:60]}")
        plt.close(fig)
        path.unlink(missing_ok=True)
        return None
    plt.close(fig)
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w_px, h_px = im.size
    w, h = w_px / 300 * 72, h_px / 300 * 72
    meta.write_text(f"{w} {h}")
    del txt
    return path, w, h


# ------------------------------------------------------------- diagrams
def _box(ax, x, y, w, h, text, fc="#eef8f7", ec="#134846", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#1f2933",
            fontweight="bold" if bold else "normal", wrap=True, linespacing=1.25)


def _arrow(ax, p, q, text=None, color="#2a8c80", fs=7.5, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, color=color, lw=1.2,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.015, text, ha="center", va="bottom", fontsize=fs,
                color="#52606d")


def _fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def draw_architecture(path: Path):
    fig, ax = _fig(9, 5.2)
    _box(ax, 0.02, 0.70, 0.22, 0.24, "Browser\n\nReact + Vite + Tailwind\n8 pages", fc="#e0f2fe", ec="#0369a1", bold=False)
    _box(ax, 0.32, 0.70, 0.30, 0.24, "FastAPI backend\nvalidation · errors · CORS\nREST API (JSON)", fc="#eef8f7")
    _box(ax, 0.70, 0.70, 0.28, 0.24, "Background jobs\ningestion · training\n(experiment suite = subprocess)", fc="#f3e8ff",
         ec="#6d28d9")
    _box(ax, 0.32, 0.36, 0.30, 0.24, "Services\ndatasets · training · anomalies\nreports · experiments", fc="#eef8f7")
    ml = "ml package\ningestion → QC → filters → windows\n→ features → reference (global /\npersonal / rolling) → detectors\n→ evaluation · explanations"
    _box(ax, 0.66, 0.30, 0.32, 0.32, ml, fc="#fef3c7", ec="#b45309", fs=8)
    _box(ax, 0.06, 0.04, 0.34, 0.22, "PostgreSQL (SQLite locally)\nmetadata · model versions · metrics\nbaselines · anomaly events · reports",
         fc="#fce7f3", ec="#be185d", fs=8)
    _box(ax, 0.52, 0.04, 0.42, 0.18, "File storage\nParquet features · 1 Hz signals · model artifacts · HTML reports",
         fc="#fce7f3", ec="#be185d", fs=8)
    _arrow(ax, (0.24, 0.82), (0.32, 0.82), "HTTPS / JSON")
    _arrow(ax, (0.62, 0.82), (0.70, 0.82), "submit")
    _arrow(ax, (0.47, 0.70), (0.47, 0.60))
    _arrow(ax, (0.84, 0.70), (0.84, 0.62))
    _arrow(ax, (0.62, 0.48), (0.66, 0.48))
    _arrow(ax, (0.40, 0.36), (0.25, 0.26))
    _arrow(ax, (0.56, 0.36), (0.68, 0.22))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_dataflow(path: Path):
    steps = [
        "Raw Empatica CSVs + tags.csv", "Parse headers: UTC start, sample rate", "Quality: missing, out-of-range, flat-line",
        "Mask invalid samples (NaN)", "Signal-specific filtering", "30 s windows, 10 s hop (common time axis)",
        "Window quality ≥ 80% valid per sensor", "14 interpretable features", "Phase labels from tags\n(window fully inside one phase)",
        "features.parquet", "Reference: global (training people) /\npersonal (own pre-stress 5 min) / rolling",
        "Detector scores + threshold", "Contributions (explanations)", "Database + dashboard + reports",
    ]
    fig, ax = _fig(9, 6.6)
    col_x = [0.03, 0.53]
    per_col = 7
    h, w, gap = 0.105, 0.43, 0.035
    pos = []
    for i, s in enumerate(steps):
        c, r = divmod(i, per_col)
        if c == 1:
            r = per_col - 1 - r  # second column flows upwards: the column jump becomes a short arrow
        x, y = col_x[c], 0.96 - (r + 1) * h - r * gap
        fc = "#fef3c7" if i in (9,) else ("#eef8f7" if c == 0 else "#e0f2fe")
        _box(ax, x, y, w, h, s, fc=fc, fs=8)
        pos.append((x, y))
    for i in range(len(steps) - 1):
        (x1, y1), (x2, y2) = pos[i], pos[i + 1]
        if i == per_col - 1:
            _arrow(ax, (x1 + w, y1 + h / 2), (x2, y2 + h / 2))
        elif i >= per_col:
            _arrow(ax, (x1 + w / 2, y1 + h), (x2 + w / 2, y2))
        else:
            _arrow(ax, (x1 + w / 2, y1), (x2 + w / 2, y2 + h))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_classes(path: Path):
    fig, ax = _fig(9, 4.6)
    items = {
        "Signal": (0.02, 0.62, "Signal\nname · start · fs\nvalues · issues"),
        "Session": (0.27, 0.62, "Session\nsignals · phases\ncalibration_end_s · excluded"),
        "Phase": (0.02, 0.18, "Phase\nname · start_s · end_s\nlabel (1 / 0 / none)"),
        "FeatureConfig": (0.27, 0.18, "FeatureConfig\nwindow 30 s · hop 10 s\nquality 80% · baseline 300 s"),
        "Detector": (0.55, 0.62, "Detector\nscorer · normalisation\nfit() · score() · contributions()\nsave() / load() (checksum)"),
        "ReferenceStats": (0.55, 0.18, "ReferenceStats\nglobal centre & scale\nwithin-person scale"),
        "Baseline": (0.80, 0.18, "ParticipantBaseline\nstatus · n_windows\ncentre · scale · period"),
        "AE": (0.80, 0.62, "AutoencoderScorer\n14→16→4→16→14\nearly stopping"),
    }
    for _, (x, y, t) in items.items():
        _box(ax, x, y, 0.19 if x < 0.5 else 0.2, 0.28, t, fs=7.6)
    _arrow(ax, (0.21, 0.76), (0.27, 0.76), "has many", style="-|>")
    _arrow(ax, (0.36, 0.62), (0.15, 0.46), "has many")
    _arrow(ax, (0.75, 0.76), (0.80, 0.76), "uses")
    _arrow(ax, (0.65, 0.62), (0.65, 0.46), "uses")
    _arrow(ax, (0.72, 0.62), (0.86, 0.46), "personal")
    _arrow(ax, (0.55, 0.70), (0.46, 0.38), "configured by")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_sequence(path: Path):
    actors = ["User\n(Monitoring)", "FastAPI", "anomalies\nservice", "Feature\nstore", "Detector\nartifact", "Database"]
    msgs = [(0, 1, "POST /api/anomalies/detect"), (1, 2, "detect()"), (2, 5, "load ModelVersion"),
            (2, 4, "Detector.load() (checksum + schema)"), (2, 3, "participant windows (incl. own baseline)"),
            (2, 4, "score() + contributions()"), (2, 5, "save DetectionRun + AnomalyEvents"),
            (2, 1, "timeline, threshold, reference"), (1, 0, "200 JSON (offline_retrospective)")]
    fig, ax = _fig(9, 5)
    xs = [0.07 + i * 0.172 for i in range(len(actors))]
    for x, a in zip(xs, actors):
        _box(ax, x - 0.07, 0.86, 0.14, 0.12, a, fs=7.6)
        ax.plot([x, x], [0.05, 0.86], color="#9aa5b1", lw=0.9, ls="--")
    for k, (a, b, t) in enumerate(msgs):
        y = 0.80 - k * 0.085
        _arrow(ax, (xs[a], y), (xs[b], y), color="#134846" if b > a else "#be185d")
        ax.text((xs[a] + xs[b]) / 2, y + 0.008, t, ha="center", va="bottom", fontsize=6.8, color="#1f2933")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


DIAGRAMS = {"architecture": (draw_architecture, "System architecture"),
            "dataflow": (draw_dataflow, "Data flow: from raw files to scores and explanations"),
            "class": (draw_classes, "Core classes of the ml package (simplified UML)"),
            "sequence": (draw_sequence, "Sequence of an offline detection request")}


def diagram_flowables(name: str, width: float) -> list:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"diagram_{name}.png"
    fn, caption = DIAGRAMS[name]
    fn(path)
    return image_flowables(path, caption, width)


def image_flowables(path: Path, caption: str, width: float) -> list:
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    disp_w = width
    disp_h = disp_w * h / w
    max_h = 17 * cm
    if disp_h > max_h:
        disp_h, disp_w = max_h, max_h * w / h
    return [KeepTogether([Image(str(path), width=disp_w, height=disp_h), Spacer(1, 3),
                          Paragraph(inline(caption), STYLES["caption"])])]


# ------------------------------------------------------------- markdown
def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_TALLEST = [0.0]  # height of the tallest inline formula in the last inline() call


def inline(text: str, fontsize: float = 9.6) -> str:
    """Convert inline Markdown (code, math, bold, italic, links) to ReportLab markup."""
    _TALLEST[0] = 0.0
    out, parts = [], re.split(r"(`[^`]+`)", text)
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            out.append(f'<font face="DejaVuMono" size="{fontsize - 1:.1f}" color="#134846">{esc(part[1:-1])}</font>')
            continue
        segs = re.split(r"(\$[^$]+\$)", part)
        for seg in segs:
            if seg.startswith("$") and seg.endswith("$") and len(seg) > 2:
                img = math_image(seg[1:-1], fontsize + 0.5)
                if img:
                    p, w, h = img
                    _TALLEST[0] = max(_TALLEST[0], h)
                    # centre the formula on the text's x-height instead of sitting on the baseline
                    out.append(f'<img src="{p.as_posix()}" width="{w:.2f}" height="{h:.2f}" '
                               f'valign="{0.35 * fontsize - h / 2:.2f}"/>')
                else:
                    out.append(f'<font face="DejaVuMono">{esc(seg[1:-1])}</font>')
                continue
            s = esc(seg)
            s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: _link(m.group(1), m.group(2)), s)
            out.append(s)
    joined = "".join(out)  # emphasis may span code spans, so apply it last
    joined = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", joined)
    joined = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", joined)
    return joined


def para(text: str, style: ParagraphStyle, **kw) -> Paragraph:
    """Paragraph whose line spacing grows when it contains tall inline formulas."""
    markup = inline(text, style.fontSize)
    if _TALLEST[0] > style.leading - 2:
        style = ParagraphStyle(f"{style.name}_tall", parent=style, leading=_TALLEST[0] + 4)
    return Paragraph(markup, style, **kw)


def _link(label: str, target: str) -> str:
    if target.startswith("http"):
        return f'<link href="{target}" color="#1f6f67"><u>{label}</u></link>'
    return f"<b>{label}</b> <font color='#7b8794'>({target.replace('../', '')})</font>"


def split_row(line: str) -> list[str]:
    """Split a Markdown table row on '|' outside $math$ and `code`."""
    cells, buf, in_math, in_code = [], "", False, False
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    for ch in s:
        if ch == "$" and not in_code:
            in_math = not in_math
        elif ch == "`":
            in_code = not in_code
        if ch == "|" and not in_math and not in_code:
            cells.append(buf.strip())
            buf = ""
        else:
            buf += ch
    cells.append(buf.strip())
    return cells


def table_flowable(rows: list[list[str]], width: float):
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    small = ncols >= 7 or width / ncols < 55
    fs = 7 if small else 8.2
    body_style = STYLES["cell_small" if small else "cell"]
    head_style = STYLES["cellh_small" if small else "cellh"]
    lengths = [max(min(len(re.sub(r"\$[^$]*\$", "xxxxxxxx", r[c])), 60) for r in rows) + 4 for c in range(ncols)]
    total = sum(lengths)

    def longest_word(c: int) -> float:
        words = re.sub(r"\$[^$]*\$|`", " ", " ".join(r[c] for r in rows)).split()
        return max((pdfmetrics.stringWidth(w, "DejaVu-Bold", fs) for w in words), default=20)

    min_w = [min(longest_word(c) + 10, width / 3) for c in range(ncols)]  # never split a word
    col_w = [max(width * l / total, m) for l, m in zip(lengths, min_w)]
    scale = width / sum(col_w)
    col_w = [w * scale for w in col_w]
    data = [[Paragraph(inline(c, fs), head_style if i == 0 else body_style) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=col_w, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BRAND),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fa")]),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def note_box(flowables: list, width: float):
    t = Table([[flowables]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NOTE_BG), ("LINEBEFORE", (0, 0), (0, -1), 3, NOTE_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def code_box(text: str, width: float):
    pre = Preformatted(text, STYLES["code"], maxLineLength=110)
    t = Table([[pre]], colWidths=[width])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f4f8")),
                           ("BOX", (0, 0), (-1, -1), 0.5, RULE), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


def mermaid_kind(src: str) -> str | None:
    s = src.strip()
    if s.startswith("classDiagram"):
        return "class"
    if s.startswith("sequenceDiagram"):
        return "sequence"
    if s.startswith("flowchart LR"):
        return "architecture"
    if s.startswith("flowchart TD"):
        return "dataflow"
    return None


class Heading(Paragraph):
    """Paragraph that registers itself in the table of contents."""

    def __init__(self, text, style, toc_level=None, key=None):
        super().__init__(text, style)
        self.toc_level, self.key = toc_level, key


def md_to_flowables(md: str, width: float, *, toc_map: dict[int, int], break_level: int | None,
                    base_dir: Path, skip_until_rule: bool = False) -> list:
    lines = md.splitlines()
    make_para = globals()["para"]  # the local list below is also called `para`
    i, story, para = 0, [], []
    heading_count = 0

    def flush_para():
        if para:
            story.append(make_para(" ".join(s.strip() for s in para), STYLES["body"]))
            para.clear()

    if skip_until_rule:  # cover page content is rendered separately
        while i < len(lines) and lines[i].strip() != "---":
            i += 1
        i += 1
    first_break_done = False
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            flush_para()
            i += 1
            continue
        if s == "---":
            flush_para()
            story.append(Spacer(1, 4))
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            flush_para()
            level = len(m.group(1))
            text = m.group(2).strip()
            if break_level is not None and level == break_level:
                if first_break_done:
                    story.append(PageBreak())
                first_break_done = True
            else:
                story.append(CondPageBreak(3.2 * cm))
            heading_count += 1
            toc_level = toc_map.get(level)
            story.append(Heading(inline(text), STYLES[f"h{level}"], toc_level, f"h{heading_count}"))
            i += 1
            continue
        if s.startswith("```"):
            flush_para()
            lang = s[3:].strip()
            block, i = [], i + 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            src = "\n".join(block)
            if lang.startswith("diagram"):
                story += diagram_flowables(lang.split()[1], width)
            elif lang == "mermaid" and mermaid_kind(src):
                story += diagram_flowables(mermaid_kind(src), width)
            else:
                story.append(code_box(src, width))
                story.append(Spacer(1, 6))
            continue
        if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
            flush_para()
            img = math_image(s[2:-2], 12)
            if img:
                p, w, h = img
                scale = min(1.0, (width * 0.9) / w)
                im = Image(str(p), width=w * scale, height=h * scale)
                story += [Spacer(1, 4), im, Spacer(1, 8)]
            else:
                story.append(code_box(s[2:-2], width))
            i += 1
            continue
        mi = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", s)
        if mi:
            flush_para()
            path = (base_dir / mi.group(2)).resolve()
            if path.exists():
                story += image_flowables(path, mi.group(1), width * 0.95)
            i += 1
            continue
        if s.startswith("|"):
            flush_para()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = split_row(lines[i])
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in row if c):
                    rows.append(row)
                i += 1
            story.append(table_flowable(rows, width))
            story.append(Spacer(1, 8))
            continue
        if s.startswith(">"):
            flush_para()
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            chunks = [p for p in "\n".join(quote).split("\n\n")]
            story.append(note_box([Paragraph(inline(" ".join(c.split("\n"))), STYLES["quote"]) for c in chunks], width))
            story.append(Spacer(1, 8))
            continue
        mb = re.match(r"^(\s*)([*-]|\d+\.)\s+(.*)$", line)
        if mb:
            flush_para()
            items = []
            while i < len(lines):
                mm_ = re.match(r"^(\s*)([*-]|\d+\.)\s+(.*)$", lines[i])
                if mm_:
                    items.append([len(mm_.group(1)), mm_.group(2), mm_.group(3)])
                    i += 1
                elif lines[i].strip() and lines[i].startswith("  ") and items:
                    items[-1][2] += " " + lines[i].strip()  # continuation line
                    i += 1
                else:
                    break
            for indent, marker, text in items:
                style = STYLES["bullet2" if indent >= 2 else "bullet"]
                bullet = "•" if marker in "*-" else marker
                story.append(make_para(text, style, bulletText=bullet))
            story.append(Spacer(1, 4))
            continue
        para.append(line)
        i += 1
    flush_para()
    return story


# --------------------------------------------------------------- document
class Doc(BaseDocTemplate):
    def __init__(self, path, running_title, **kw):
        super().__init__(str(path), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=2.2 * cm,
                         bottomMargin=2 * cm, title=running_title, author="MOSAIC project", **kw)
        self.running_title = running_title
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("cover", [frame], onPage=lambda c, d: None),
                               PageTemplate("normal", [frame], onPage=self._decorate)])

    def _decorate(self, canv, doc):
        canv.saveState()
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.6)
        canv.line(self.leftMargin, A4[1] - 1.5 * cm, A4[0] - self.rightMargin, A4[1] - 1.5 * cm)
        canv.setFont("DejaVu", 7.8)
        canv.setFillColor(SOFT)
        canv.drawString(self.leftMargin, A4[1] - 1.3 * cm, self.running_title)
        canv.drawRightString(A4[0] - self.rightMargin, A4[1] - 1.3 * cm, "MOSAIC · research use only")
        canv.drawCentredString(A4[0] / 2, 1.1 * cm, f"{doc.page}")
        canv.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Heading) and flowable.toc_level is not None:
            text = re.sub(r"<[^>]+>", "", flowable.getPlainText())
            self.canv.bookmarkPage(flowable.key)
            self.canv.addOutlineEntry(text, flowable.key, level=flowable.toc_level, closed=flowable.toc_level > 0)
            self.notify("TOCEntry", (flowable.toc_level, text, self.page, flowable.key))


def cover(title: str, subtitle: str, lines: list[str], badge: str) -> list:
    from reportlab.platypus import NextPageTemplate
    story = [Spacer(1, 3.2 * cm)]
    story.append(Paragraph(f'<font color="#2a8c80" size="10">{badge}</font>', STYLES["body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(title, ParagraphStyle("ct", parent=BASE, fontName="DejaVu-Bold", fontSize=30, leading=36,
                                                 textColor=BRAND)))
    story.append(Spacer(1, 6))
    story.append(Paragraph(subtitle, ParagraphStyle("cs", parent=BASE, fontSize=14, leading=19, textColor=INK)))
    story.append(Spacer(1, 1.2 * cm))
    for ln in lines:
        story.append(Paragraph(ln, ParagraphStyle("cl", parent=BASE, fontSize=10.5, leading=16, textColor=SOFT)))
    story.append(NextPageTemplate("normal"))
    story.append(PageBreak())
    return story


def toc_section() -> list:
    toc = TableOfContents()
    toc.levelStyles = [STYLES["toc0"], STYLES["toc1"]]
    toc.dotsMinLevel = 0
    return [Paragraph("Contents", STYLES["h1"]), toc, PageBreak()]


def build_report():
    md = (DOCS / "report.md").read_text(encoding="utf-8")
    path = OUT / "MOSAIC_Project_Report.pdf"
    doc = Doc(path, "MOSAIC: Project Report")
    width = doc.width
    story = cover(
        "MOSAIC",
        "Multimodal Observation System for Anomaly Identification and Clinical Trends<br/>"
        "<font size='11' color='#52606d'>Personalised versus global anomaly detection in multimodal wearable "
        "physiological signals</font>",
        ["Major project report: B.Tech Computer Science and Engineering (Data Science)",
         "Student: <i>[Name, Roll number]</i> &nbsp;&nbsp;·&nbsp;&nbsp; Supervisor: <i>[Name, Designation]</i>",
         "Institution: <i>[Institution name]</i> &nbsp;&nbsp;·&nbsp;&nbsp; Academic year: <i>[2025–2026]</i>",
         "<br/>Dataset: PhysioNet Wearable Device Dataset from Induced Stress and Structured Exercise Sessions v1.0.1 "
         "(DOI 10.13026/he0v-tf17)",
         "Results: experiment run <font face='DejaVuMono'>exp_20261015T064235Z</font> "
         "(reports/experiments/main/results.json)",
         f"Generated {date.today().strftime('%d %B %Y')}",
         "<br/><b>Research use only.</b> Not a medical device; not for diagnosis."],
        "FINAL-YEAR MAJOR PROJECT REPORT")
    story += toc_section()
    body = md_to_flowables(md, width, toc_map={2: 0, 3: 1}, break_level=None, base_dir=DOCS, skip_until_rule=True)
    story += body
    doc.multiBuild(story)
    return path


def build_guide():
    md = (DOCS / "learning_guide.md").read_text(encoding="utf-8")
    path = OUT / "MOSAIC_Learning_Guide.pdf"
    doc = Doc(path, "MOSAIC: Learning and Explanation Guide")
    width = doc.width
    story = cover(
        "Understanding MOSAIC",
        "A learning and explanation guide<br/><font size='11' color='#52606d'>Everything you need to understand, "
        "present and defend the project, from sensors and statistics to results and viva questions</font>",
        ["MOSAIC: Multimodal Observation System for Anomaly Identification and Clinical Trends",
         "B.Tech CSE (Data Science) major project",
         "<br/>All numbers come from the real experiment run "
         "<font face='DejaVuMono'>exp_20261015T064235Z</font> on the PhysioNet wearable dataset.",
         f"Generated {date.today().strftime('%d %B %Y')}"],
        "STUDY GUIDE")
    story += toc_section()
    # Drop the title and subtitle lines (the cover shows them); keep the "how to use" note.
    md = "\n".join(md.splitlines()[3:])
    body = md_to_flowables(md, width, toc_map={1: 0, 2: 1}, break_level=1, base_dir=DOCS)
    story += body
    doc.multiBuild(story)
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (build_report, build_guide):
        p = fn()
        from pypdf import PdfReader
        print(f"wrote {p.relative_to(ROOT)}  ({len(PdfReader(str(p)).pages)} pages, {p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    sys.exit(main())
