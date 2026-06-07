"""Small dependency-free SVG chart helpers.

The project already has a pandas/scikit-learn path, but the verified training
path is intentionally stdlib-only. These helpers keep plotting available even
when matplotlib is not installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import html
import math

FONT = "'Noto Sans CJK SC','Microsoft YaHei','PingFang SC','WenQuanYi Micro Hei',Arial,sans-serif"
BG = "#f8fafc"
PANEL = "#ffffff"
INK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"
BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#f59e0b"
PURPLE = "#7c3aed"
CYAN = "#0891b2"
PALETTE = [BLUE, GREEN, AMBER, PURPLE, CYAN, "#db2777", "#475569", "#059669"]


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        v = float(value)
    except Exception:  # noqa: BLE001
        return default
    return v if math.isfinite(v) else default


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _polar(cx: float, cy: float, r: float, angle_deg: float) -> tuple[float, float]:
    rad = math.radians(angle_deg)
    return cx + r * math.cos(rad), cy + r * math.sin(rad)


def _svg_root(width: int, height: int, body: str) -> str:
    return f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\" role=\"img\">
<style>
  text {{ font-family: {FONT}; fill: {INK}; }}
  .title {{ font-size: 24px; font-weight: 800; }}
  .subtitle {{ font-size: 13px; fill: {MUTED}; }}
  .label {{ font-size: 12px; fill: {INK}; }}
  .muted {{ font-size: 11px; fill: {MUTED}; }}
  .axis {{ stroke: {GRID}; stroke-width: 1; }}
</style>
<rect width=\"100%\" height=\"100%\" fill=\"{BG}\"/>
<rect x=\"14\" y=\"14\" width=\"{width - 28}\" height=\"{height - 28}\" rx=\"18\" fill=\"{PANEL}\"/>
{body}
</svg>
"""


def write_horizontal_bar_chart(
    path: Path,
    *,
    title: str,
    subtitle: str,
    rows: list[tuple[str, float]],
    width: int = 1120,
    row_height: int = 34,
    value_kind: str = "pct",
    positive_color: str = BLUE,
    negative_color: str = RED,
) -> Path:
    """Write a horizontal bar chart that supports positive/negative values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(str(label), float(value)) for label, value in rows if _finite(value) is not None]
    if not rows:
        rows = [("无可制图数据", 0.0)]
    height = max(260, 132 + len(rows) * row_height)
    left = 270
    right = width - 120
    top = 96
    chart_w = right - left
    values = [v for _, v in rows]
    min_v = min(values + [0.0])
    max_v = max(values + [0.0])
    if abs(max_v - min_v) < 1e-12:
        max_v += 1.0
        min_v -= 1.0
    pad = (max_v - min_v) * 0.08
    min_v -= pad
    max_v += pad

    def x_for(v: float) -> float:
        return left + (v - min_v) / (max_v - min_v) * chart_w

    zero_x = x_for(0.0)
    elems = [
        f"<text x=\"42\" y=\"52\" class=\"title\">{_esc(title)}</text>",
        f"<text x=\"42\" y=\"76\" class=\"subtitle\">{_esc(subtitle)}</text>",
        f"<line x1=\"{zero_x:.1f}\" y1=\"{top - 16}\" x2=\"{zero_x:.1f}\" y2=\"{top + len(rows) * row_height + 10}\" stroke=\"{MUTED}\" stroke-width=\"1.2\" stroke-dasharray=\"4 4\"/>",
    ]
    for tick in (min_v, 0.0, max_v):
        x = x_for(tick)
        elems.append(f"<line x1=\"{x:.1f}\" y1=\"{top - 14}\" x2=\"{x:.1f}\" y2=\"{top + len(rows) * row_height + 8}\" class=\"axis\"/>")
        tick_text = _fmt_pct(tick) if value_kind == "pct" else f"{tick:.3f}"
        elems.append(f"<text x=\"{x:.1f}\" y=\"{top + len(rows) * row_height + 30}\" text-anchor=\"middle\" class=\"muted\">{tick_text}</text>")
    for i, (label, value) in enumerate(rows):
        y = top + i * row_height
        x0 = min(zero_x, x_for(value))
        bar_w = max(1.0, abs(x_for(value) - zero_x))
        color = positive_color if value >= 0 else negative_color
        val_text = _fmt_pct(value) if value_kind == "pct" else f"{value:.4f}"
        elems.append(f"<text x=\"42\" y=\"{y + 20}\" class=\"label\">{_esc(label[:34])}</text>")
        elems.append(f"<rect x=\"{x0:.1f}\" y=\"{y + 5}\" width=\"{bar_w:.1f}\" height=\"18\" rx=\"5\" fill=\"{color}\" opacity=\"0.88\"/>")
        text_x = x_for(value) + (7 if value >= 0 else -7)
        anchor = "start" if value >= 0 else "end"
        elems.append(f"<text x=\"{text_x:.1f}\" y=\"{y + 20}\" text-anchor=\"{anchor}\" class=\"label\">{val_text}</text>")
    path.write_text(_svg_root(width, height, "\n".join(elems)), encoding="utf-8")
    return path


def write_column_chart(
    path: Path,
    *,
    title: str,
    subtitle: str,
    rows: list[tuple[str, float]],
    width: int = 980,
    height: int = 430,
    value_kind: str = "number",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(str(label), float(value)) for label, value in rows if _finite(value) is not None]
    if not rows:
        rows = [("无数据", 0.0)]
    left, right, top, bottom = 70, width - 38, 96, height - 76
    chart_w, chart_h = right - left, bottom - top
    values = [v for _, v in rows]
    min_v = min(values + [0.0])
    max_v = max(values + [0.0])
    if max_v <= min_v:
        max_v = min_v + 1.0
    pad = (max_v - min_v) * 0.12
    min_v = min(0.0, min_v - pad)
    max_v += pad

    def y_for(v: float) -> float:
        return bottom - (v - min_v) / (max_v - min_v) * chart_h

    elems = [
        f"<text x=\"42\" y=\"52\" class=\"title\">{_esc(title)}</text>",
        f"<text x=\"42\" y=\"76\" class=\"subtitle\">{_esc(subtitle)}</text>",
    ]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = min_v + (max_v - min_v) * frac
        y = y_for(v)
        elems.append(f"<line x1=\"{left}\" y1=\"{y:.1f}\" x2=\"{right}\" y2=\"{y:.1f}\" class=\"axis\"/>")
        label = _fmt_pct(v) if value_kind == "pct" else f"{v:.2f}"
        elems.append(f"<text x=\"{left - 10}\" y=\"{y + 4:.1f}\" text-anchor=\"end\" class=\"muted\">{label}</text>")
    gap = 12
    bar_w = max(16, (chart_w - gap * (len(rows) + 1)) / len(rows))
    zero_y = y_for(0.0)
    elems.append(f"<line x1=\"{left}\" y1=\"{zero_y:.1f}\" x2=\"{right}\" y2=\"{zero_y:.1f}\" stroke=\"{MUTED}\" stroke-width=\"1.2\"/>")
    for i, (label, value) in enumerate(rows):
        x = left + gap + i * (bar_w + gap)
        y = y_for(max(value, 0.0))
        h = max(1.0, abs(y_for(value) - zero_y))
        if value < 0:
            y = zero_y
        color = PALETTE[i % len(PALETTE)] if value >= 0 else RED
        val_text = _fmt_pct(value) if value_kind == "pct" else f"{value:.0f}"
        elems.append(f"<rect x=\"{x:.1f}\" y=\"{y:.1f}\" width=\"{bar_w:.1f}\" height=\"{h:.1f}\" rx=\"6\" fill=\"{color}\" opacity=\"0.88\"/>")
        elems.append(f"<text x=\"{x + bar_w / 2:.1f}\" y=\"{min(y, zero_y) - 7:.1f}\" text-anchor=\"middle\" class=\"muted\">{val_text}</text>")
        elems.append(f"<text x=\"{x + bar_w / 2:.1f}\" y=\"{bottom + 22}\" text-anchor=\"middle\" class=\"muted\">{_esc(label[:12])}</text>")
    path.write_text(_svg_root(width, height, "\n".join(elems)), encoding="utf-8")
    return path


def write_pie_chart(
    path: Path,
    *,
    title: str,
    subtitle: str,
    rows: list[tuple[str, float]],
    width: int = 980,
    height: int = 560,
    add_remainder: bool = True,
    remainder_label: str = "其他/未披露",
) -> Path:
    """Write a dependency-free SVG pie chart.

    ``rows`` values should be fractions (0.25 means 25%).  When
    ``add_remainder`` is true and the provided ratios sum to less than 100%, an
    extra remainder slice is added so shareholder concentration is visually
    comparable across companies.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    clean: list[tuple[str, float]] = []
    for label, value in rows:
        v = _finite(value)
        if v is None or v <= 0:
            continue
        clean.append((str(label), min(float(v), 1.0)))
    total = sum(v for _, v in clean)
    if add_remainder and total < 0.999:
        clean.append((remainder_label, max(0.0, 1.0 - total)))
        total = 1.0
    if not clean or total <= 0:
        clean = [("无数据", 1.0)]
        total = 1.0

    cx, cy, r = 295, 305, 168
    elems = [
        f"<text x=\"42\" y=\"52\" class=\"title\">{_esc(title)}</text>",
        f"<text x=\"42\" y=\"76\" class=\"subtitle\">{_esc(subtitle)}</text>",
    ]
    angle = -90.0
    for i, (label, value) in enumerate(clean):
        frac = value / total
        sweep = frac * 360.0
        color = PALETTE[i % len(PALETTE)]
        if sweep >= 359.999:
            elems.append(f"<circle cx=\"{cx}\" cy=\"{cy}\" r=\"{r}\" fill=\"{color}\" opacity=\"0.9\"/>")
        else:
            x1, y1 = _polar(cx, cy, r, angle)
            x2, y2 = _polar(cx, cy, r, angle + sweep)
            large = 1 if sweep > 180 else 0
            elems.append(
                f"<path d=\"M {cx:.1f} {cy:.1f} L {x1:.1f} {y1:.1f} "
                f"A {r:.1f} {r:.1f} 0 {large} 1 {x2:.1f} {y2:.1f} Z\" "
                f"fill=\"{color}\" opacity=\"0.9\" stroke=\"{PANEL}\" stroke-width=\"2\"/>"
            )
        mid = angle + sweep / 2
        if frac >= 0.055:
            tx, ty = _polar(cx, cy, r * 0.62, mid)
            elems.append(
                f"<text x=\"{tx:.1f}\" y=\"{ty:.1f}\" text-anchor=\"middle\" class=\"label\" "
                f"font-weight=\"700\">{_fmt_pct(value)}</text>"
            )
        angle += sweep

    legend_x, legend_y = 520, 118
    row_h = 34
    for i, (label, value) in enumerate(clean):
        y = legend_y + i * row_h
        color = PALETTE[i % len(PALETTE)]
        elems.append(f"<rect x=\"{legend_x}\" y=\"{y - 12}\" width=\"18\" height=\"18\" rx=\"4\" fill=\"{color}\" opacity=\"0.9\"/>")
        elems.append(f"<text x=\"{legend_x + 28}\" y=\"{y + 2}\" class=\"label\">{_esc(label[:24])}</text>")
        elems.append(f"<text x=\"{width - 54}\" y=\"{y + 2}\" text-anchor=\"end\" class=\"label\">{_fmt_pct(value)}</text>")
    path.write_text(_svg_root(width, height, "\n".join(elems)), encoding="utf-8")
    return path


def write_line_chart(
    path: Path,
    *,
    title: str,
    subtitle: str,
    series: dict[str, list[tuple[str, float]]],
    width: int = 1120,
    height: int = 520,
    value_kind: str = "pct",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    series = {name: [(str(x), float(y)) for x, y in points if _finite(y) is not None] for name, points in series.items()}
    series = {name: points for name, points in series.items() if points}
    if not series:
        series = {"无数据": [("", 0.0)]}
    left, right, top, bottom = 82, width - 48, 102, height - 84
    chart_w, chart_h = right - left, bottom - top
    xs = sorted({x for points in series.values() for x, _ in points})
    x_index = {x: i for i, x in enumerate(xs)}
    values = [y for points in series.values() for _, y in points]
    min_v, max_v = min(values), max(values)
    if abs(max_v - min_v) < 1e-12:
        max_v += 1.0
        min_v -= 1.0
    pad = (max_v - min_v) * 0.12
    min_v -= pad
    max_v += pad

    def x_for(label: str) -> float:
        if len(xs) == 1:
            return left + chart_w / 2
        return left + x_index[label] / (len(xs) - 1) * chart_w

    def y_for(v: float) -> float:
        return bottom - (v - min_v) / (max_v - min_v) * chart_h

    elems = [
        f"<text x=\"42\" y=\"52\" class=\"title\">{_esc(title)}</text>",
        f"<text x=\"42\" y=\"76\" class=\"subtitle\">{_esc(subtitle)}</text>",
    ]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = min_v + (max_v - min_v) * frac
        y = y_for(v)
        elems.append(f"<line x1=\"{left}\" y1=\"{y:.1f}\" x2=\"{right}\" y2=\"{y:.1f}\" class=\"axis\"/>")
        label = _fmt_pct(v) if value_kind == "pct" else f"{v:.2f}"
        elems.append(f"<text x=\"{left - 10}\" y=\"{y + 4:.1f}\" text-anchor=\"end\" class=\"muted\">{label}</text>")
    if min_v <= 0 <= max_v:
        zero_y = y_for(0.0)
        elems.append(f"<line x1=\"{left}\" y1=\"{zero_y:.1f}\" x2=\"{right}\" y2=\"{zero_y:.1f}\" stroke=\"{MUTED}\" stroke-width=\"1.2\" stroke-dasharray=\"4 4\"/>")
    tick_labels = [xs[0], xs[len(xs) // 2], xs[-1]] if len(xs) > 2 else xs
    for label in dict.fromkeys(tick_labels):
        x = x_for(label)
        elems.append(f"<text x=\"{x:.1f}\" y=\"{bottom + 28}\" text-anchor=\"middle\" class=\"muted\">{_esc(label)}</text>")
    for i, (name, points) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
        poly = " ".join(f"{x_for(x):.1f},{y_for(y):.1f}" for x, y in points)
        elems.append(f"<polyline points=\"{poly}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2.6\" stroke-linejoin=\"round\" stroke-linecap=\"round\"/>")
        lx = left + i * 170
        elems.append(f"<rect x=\"{lx}\" y=\"{height - 42}\" width=\"18\" height=\"5\" rx=\"2\" fill=\"{color}\"/>")
        elems.append(f"<text x=\"{lx + 26}\" y=\"{height - 36}\" class=\"muted\">{_esc(name)}</text>")
    path.write_text(_svg_root(width, height, "\n".join(elems)), encoding="utf-8")
    return path
