#!/usr/bin/env python3
"""Generate the vector results summary figure used by paper.tex.

Requires ReportLab. The figure deliberately avoids the historical phrase
"cache hit rate": all cache panels report cached prompt tokens divided by
total prompt tokens.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import EmbeddedType1Face, Font
from reportlab.pdfgen import canvas


BLUE = HexColor("#3465AD")
PURPLE = HexColor("#6A4C93")
LIGHT_PURPLE = HexColor("#A995CE")
PALE_PURPLE = HexColor("#EEE9F6")
GRID = HexColor("#D9D9D9")
TEXT = HexColor("#202020")
MUTED = HexColor("#555555")
RED = HexColor("#C94455")
REGULAR = "FigureRoman"
BOLD = "FigureRomanBold"

WIDTH = 7.16 * 72
HEIGHT = 2.35 * 72


def _register_fonts() -> None:
    """Embed the same URW Times Type 1 family used by IEEEtran."""

    def locate(name: str) -> str:
        result = subprocess.run(
            ["kpsewhich", name], check=True, capture_output=True, text=True
        )
        path = result.stdout.strip()
        if not path:
            raise RuntimeError(f"TeX font not found: {name}")
        return path

    regular_face = EmbeddedType1Face(locate("utmr8a.afm"), locate("utmr8a.pfb"))
    bold_face = EmbeddedType1Face(locate("utmb8a.afm"), locate("utmb8a.pfb"))
    pdfmetrics.registerTypeFace(regular_face)
    pdfmetrics.registerTypeFace(bold_face)
    pdfmetrics.registerFont(Font(REGULAR, regular_face.name, "WinAnsiEncoding"))
    pdfmetrics.registerFont(Font(BOLD, bold_face.name, "WinAnsiEncoding"))


def _x(value: float, minimum: float, maximum: float, left: float, right: float) -> float:
    return left + (value - minimum) / (maximum - minimum) * (right - left)


def _y(value: float, minimum: float, maximum: float, bottom: float, top: float) -> float:
    return bottom + (value - minimum) / (maximum - minimum) * (top - bottom)


def _centered(c: canvas.Canvas, x: float, y: float, text: str, size: float = 6.5) -> None:
    c.setFont(REGULAR, size)
    c.setFillColor(TEXT)
    c.drawCentredString(x, y, text)


def _panel_axes(
    c: canvas.Canvas,
    left: float,
    right: float,
    bottom: float,
    top: float,
    y_min: float,
    y_max: float,
    y_ticks: list[float],
    y_label: str,
) -> None:
    c.setLineWidth(0.65)
    c.setStrokeColor(TEXT)
    c.line(left, bottom, right, bottom)
    c.line(left, bottom, left, top)
    for tick in y_ticks:
        yy = _y(tick, y_min, y_max, bottom, top)
        c.setStrokeColor(GRID)
        c.setLineWidth(0.35)
        c.line(left, yy, right, yy)
        c.setFillColor(TEXT)
        c.setFont(REGULAR, 6.1)
        c.drawRightString(left - 3.5, yy - 2.1, f"{tick:g}")
    c.saveState()
    c.translate(left - 26, (bottom + top) / 2)
    c.rotate(90)
    _centered(c, 0, -2.3, y_label, 6.5)
    c.restoreState()


def _error_bar(
    c: canvas.Canvas,
    x: float,
    mean: float,
    sd: float,
    y_min: float,
    y_max: float,
    bottom: float,
    top: float,
    colour,
) -> None:
    if sd <= 0:
        return
    lo = _y(mean - sd, y_min, y_max, bottom, top)
    hi = _y(mean + sd, y_min, y_max, bottom, top)
    c.setStrokeColor(colour)
    c.setLineWidth(0.7)
    c.line(x, lo, x, hi)
    c.line(x - 2.2, lo, x + 2.2, lo)
    c.line(x - 2.2, hi, x + 2.2, hi)


def _series(
    c: canvas.Canvas,
    xs: list[float],
    ys: list[float],
    sds: list[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    right: float,
    bottom: float,
    top: float,
    colour,
    triangle: bool,
) -> None:
    points = [
        (_x(x, x_min, x_max, left, right), _y(y, y_min, y_max, bottom, top))
        for x, y in zip(xs, ys)
    ]
    c.setStrokeColor(colour)
    c.setLineWidth(1.45)
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        c.line(x1, y1, x2, y2)
    for (xx, yy), mean, sd in zip(points, ys, sds):
        _error_bar(c, xx, mean, sd, y_min, y_max, bottom, top, colour)
        c.setFillColor(colour)
        c.setStrokeColor(colour)
        if triangle:
            p = c.beginPath()
            p.moveTo(xx, yy + 3.2)
            p.lineTo(xx - 3.0, yy - 2.5)
            p.lineTo(xx + 3.0, yy - 2.5)
            p.close()
            c.drawPath(p, fill=1, stroke=0)
        else:
            c.circle(xx, yy, 2.5, fill=1, stroke=0)


def _shared_legend(c: canvas.Canvas) -> None:
    y = HEIGHT - 7.8
    x1 = WIDTH / 2 - 90
    c.setStrokeColor(BLUE)
    c.setFillColor(BLUE)
    c.setLineWidth(1.4)
    c.line(x1, y, x1 + 14, y)
    c.circle(x1 + 7, y, 2.4, fill=1, stroke=0)
    c.setFillColor(TEXT)
    c.setFont(REGULAR, 6.7)
    c.drawString(x1 + 18, y - 2.2, "Flat cache-aware")

    x2 = WIDTH / 2 + 15
    c.setStrokeColor(PURPLE)
    c.setFillColor(PURPLE)
    c.line(x2, y, x2 + 14, y)
    p = c.beginPath()
    p.moveTo(x2 + 7, y + 3.0)
    p.lineTo(x2 + 4, y - 2.5)
    p.lineTo(x2 + 10, y - 2.5)
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    c.setFillColor(TEXT)
    c.drawString(x2 + 18, y - 2.2, "Per-worker tree")


def build_figure(output: Path) -> None:
    _register_fonts()
    output.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(
        str(output),
        pagesize=(WIDTH, HEIGHT),
        pageCompression=1,
        initialFontName=REGULAR,
        initialFontSize=7,
    )
    c.setTitle("Worker-specific routing: load and locality characterisation")
    c.setFillColor(TEXT)
    _shared_legend(c)

    bottom, top = 34.0, 137.0
    panels = [(34.0, 178.0), (205.0, 349.0), (382.0, 508.0)]
    loads = [4.0, 8.0, 12.0]

    # Panel (a): aggregate cached-token fraction.
    left, right = panels[0]
    c.setFont(BOLD, 7.7)
    c.drawString(left, 146.0, "(a) Cache reuse vs. load")
    _panel_axes(c, left, right, bottom, top, 60, 80, [60, 65, 70, 75, 80], "Cached-token fraction (%)")
    ca_cached = [66.0, 66.4, 64.6]
    pwt_cached = [76.2, 76.1, 76.6]
    ca_cached_sd = [0.0, 0.45, 0.50]
    pwt_cached_sd = [0.0, 0.06, 0.31]
    gap_path = c.beginPath()
    for i, (x, y) in enumerate(zip(loads, ca_cached)):
        xx = _x(x, 4, 12, left, right)
        yy = _y(y, 60, 80, bottom, top)
        gap_path.moveTo(xx, yy) if i == 0 else gap_path.lineTo(xx, yy)
    for x, y in reversed(list(zip(loads, pwt_cached))):
        gap_path.lineTo(_x(x, 4, 12, left, right), _y(y, 60, 80, bottom, top))
    gap_path.close()
    c.setFillColor(PALE_PURPLE)
    c.drawPath(gap_path, fill=1, stroke=0)
    _series(c, loads, ca_cached, ca_cached_sd, 4, 12, 60, 80, left, right, bottom, top, BLUE, False)
    _series(c, loads, pwt_cached, pwt_cached_sd, 4, 12, 60, 80, left, right, bottom, top, PURPLE, True)
    for x, gap, y in zip(loads, [10.2, 9.7, 12.0], pwt_cached):
        c.setFillColor(PURPLE)
        c.setFont(BOLD, 6.0)
        c.drawCentredString(_x(x, 4, 12, left, right), _y(y, 60, 80, bottom, top) + 7, f"+{gap:.1f} pp")
    # Panel (b): completion-path TTFT in the historical runs.
    left, right = panels[1]
    c.setFillColor(TEXT)
    c.setFont(BOLD, 7.7)
    c.drawString(left, 146.0, "(b) Recorded TTFT p50")
    _panel_axes(c, left, right, bottom, top, 0, 3.0, [0, 0.75, 1.5, 2.25, 3.0], "TTFT p50 (s)")
    ca_ttft = [0.094, 0.278, 2.411]
    pwt_ttft = [0.072, 0.147, 0.226]
    ca_ttft_sd = [0.0, 0.028, 0.520]
    pwt_ttft_sd = [0.0, 0.001, 0.014]
    _series(c, loads, ca_ttft, ca_ttft_sd, 4, 12, 0, 3.0, left, right, bottom, top, BLUE, False)
    _series(c, loads, pwt_ttft, pwt_ttft_sd, 4, 12, 0, 3.0, left, right, bottom, top, PURPLE, True)
    c.setFillColor(RED)
    c.setFont(BOLD, 6.4)
    c.drawCentredString(_x(12, 4, 12, left, right) - 7, _y(2.72, 0, 3.0, bottom, top), "10.7x lower*")

    # Panel (c): locality contraction at the replicated 12x point.
    left, right = panels[2]
    c.setFillColor(TEXT)
    c.setFont(BOLD, 7.7)
    c.drawString(left, 146.0, "(c) Locality dependence")
    _panel_axes(c, left, right, bottom, top, 0, 14, [0, 4, 8, 12], "Per-worker-tree gain (pp)")
    centres = [left + (right - left) * 0.28, left + (right - left) * 0.75]
    gaps = [12.0, 4.2]
    colours = [PURPLE, LIGHT_PURPLE]
    width = 31.0
    for xx, value, colour in zip(centres, gaps, colours):
        yy = _y(value, 0, 14, bottom, top)
        c.setFillColor(colour)
        c.rect(xx - width / 2, bottom, width, yy - bottom, fill=1, stroke=0)
        c.setFillColor(TEXT)
        c.setFont(BOLD, 6.4)
        c.drawCentredString(xx, yy + 4, f"{value:.1f} pp")
    c.setFillColor(TEXT)
    c.setFont(REGULAR, 6.0)
    c.drawCentredString(centres[0], 23.0, "High locality")
    c.drawCentredString(centres[1], 23.0, "Low locality")
    c.setFillColor(MUTED)
    c.setFont(REGULAR, 5.5)
    c.drawCentredString(centres[0], 15.0, "Zipf 1.5, session 4")
    c.drawCentredString(centres[1], 15.0, "Zipf 0.7, session 2")

    for left, right in panels[:2]:
        for load in loads:
            xx = _x(load, 4, 12, left, right)
            c.setStrokeColor(TEXT)
            c.setLineWidth(0.55)
            c.line(xx, bottom, xx, bottom - 3)
            _centered(c, xx, 22.0, f"{int(load)}x", 6.2)
        _centered(c, (left + right) / 2, 10.0, "Offered-load speedup", 6.4)

    c.showPage()
    c.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("result_summary.pdf"),
        help="output PDF path",
    )
    args = parser.parse_args()
    build_figure(args.output)


if __name__ == "__main__":
    main()
