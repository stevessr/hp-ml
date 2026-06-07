"""matplotlib-based chart generation for reports.

All charts now use matplotlib with a consistent style inspired by the original
hand-written SVG aesthetic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import numpy as np

# Configure matplotlib for Chinese fonts and consistent styling
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "PingFang SC",
    "WenQuanYi Micro Hei",
    "SimHei",
    "Arial",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

# Color scheme matching original SVG design
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


def _finite(value: Any, default: float | None = None) -> float | None:
    """Convert value to finite float or return default."""
    try:
        v = float(value)
    except Exception:  # noqa: BLE001
        return default
    return v if math.isfinite(v) else default


def _fmt_pct(value: float) -> str:
    """Format value as percentage."""
    return f"{value * 100:.2f}%"


def _apply_style(fig: Figure, ax: plt.Axes) -> None:
    """Apply consistent styling to figure and axes."""
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8, color=GRID)


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
    """Generate horizontal bar chart using matplotlib."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Clean and validate data
    rows = [(str(label), float(value)) for label, value in rows if _finite(value) is not None]
    if not rows:
        rows = [("无可制图数据", 0.0)]

    # Calculate figure size
    fig_height = max(4, 2 + len(rows) * row_height / 72)  # Convert pixels to inches
    fig_width = width / 72

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    _apply_style(fig, ax)

    # Extract labels and values
    labels = [label[:34] for label, _ in rows]
    values = [value for _, value in rows]

    # Create bar colors based on positive/negative
    colors = [positive_color if v >= 0 else negative_color for v in values]

    # Create horizontal bars
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.88, height=0.7)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        if value_kind == "pct":
            label_text = _fmt_pct(value)
        else:
            label_text = f"{value:.4f}"

        x_pos = value + (0.01 * (max(values) - min(values)) if value >= 0 else -0.01 * (max(values) - min(values)))
        ha = "left" if value >= 0 else "right"
        ax.text(x_pos, i, label_text, va="center", ha=ha, fontsize=10, color=INK, fontweight="bold")

    # Set labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()  # Top to bottom

    # Format x-axis
    if value_kind == "pct":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x*100:.1f}%"))

    # Add zero line
    ax.axvline(x=0, color=MUTED, linestyle="--", linewidth=1.5, alpha=0.8)

    # Add title and subtitle
    fig.suptitle(title, fontsize=16, fontweight="bold", color=INK, x=0.08, y=0.97, ha="left")
    ax.text(0.08, 0.93, subtitle, transform=fig.transFigure, fontsize=11, color=MUTED, ha="left")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

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
    """Generate vertical column chart using matplotlib."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Clean and validate data
    rows = [(str(label), float(value)) for label, value in rows if _finite(value) is not None]
    if not rows:
        rows = [("无数据", 0.0)]

    fig_width = width / 72
    fig_height = height / 72

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    _apply_style(fig, ax)

    # Extract labels and values
    labels = [label[:12] for label, _ in rows]
    values = [value for _, value in rows]

    # Create color map
    colors = [PALETTE[i % len(PALETTE)] if v >= 0 else RED for i, v in enumerate(values)]

    # Create column bars
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, values, color=colors, alpha=0.88, width=0.7)

    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if value_kind == "pct":
            label_text = _fmt_pct(value)
        else:
            label_text = f"{value:.0f}"

        y_pos = height if height > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, label_text,
                ha="center", va="bottom" if height >= 0 else "top",
                fontsize=9, color=MUTED, fontweight="bold")

    # Set labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9, rotation=0)

    # Format y-axis
    if value_kind == "pct":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"{y*100:.1f}%"))

    # Add zero line
    ax.axhline(y=0, color=MUTED, linestyle="-", linewidth=1.5, alpha=0.8)

    # Add title and subtitle
    fig.suptitle(title, fontsize=16, fontweight="bold", color=INK, x=0.08, y=0.96, ha="left")
    ax.text(0.08, 0.91, subtitle, transform=fig.transFigure, fontsize=11, color=MUTED, ha="left")

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

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
    """Generate pie chart using matplotlib.

    ``rows`` values should be fractions (0.25 means 25%).  When
    ``add_remainder`` is true and the provided ratios sum to less than 100%, an
    extra remainder slice is added so shareholder concentration is visually
    comparable across companies.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Clean and validate data
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

    fig_width = width / 72
    fig_height = height / 72

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    # Extract labels and values
    labels = [label[:24] for label, _ in clean]
    values = [value for _, value in clean]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(clean))]

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        colors=colors,
        autopct=lambda pct: f"{pct:.2f}%" if pct >= 5.5 else "",
        startangle=90,
        counterclock=False,
        wedgeprops={"alpha": 0.9, "edgecolor": PANEL, "linewidth": 2},
        textprops={"fontsize": 11, "fontweight": "bold", "color": "white"}
    )

    # Create legend
    legend_labels = [f"{label}  {_fmt_pct(value)}" for label, value in clean]
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=10,
        frameon=False,
    )

    # Add title and subtitle
    fig.suptitle(title, fontsize=16, fontweight="bold", color=INK, x=0.08, y=0.96, ha="left")
    ax.text(0.08, 0.91, subtitle, transform=fig.transFigure, fontsize=11, color=MUTED, ha="left")

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

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
    """Generate line chart using matplotlib."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Clean and validate data
    series = {
        name: [(str(x), float(y)) for x, y in points if _finite(y) is not None]
        for name, points in series.items()
    }
    series = {name: points for name, points in series.items() if points}

    if not series:
        series = {"无数据": [("", 0.0)]}

    fig_width = width / 72
    fig_height = height / 72

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    _apply_style(fig, ax)

    # Plot each series
    for i, (name, points) in enumerate(series.items()):
        x_labels = [x for x, _ in points]
        y_values = [y for _, y in points]
        color = PALETTE[i % len(PALETTE)]

        ax.plot(
            range(len(x_labels)),
            y_values,
            color=color,
            linewidth=2.6,
            marker="o",
            markersize=5,
            label=name,
            alpha=0.9,
        )

    # Set x-axis labels
    if series:
        first_series = list(series.values())[0]
        x_labels = [x for x, _ in first_series]
        x_pos = range(len(x_labels))
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9, rotation=45, ha="right")

    # Format y-axis
    if value_kind == "pct":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"{y*100:.1f}%"))

    # Add zero line if data crosses zero
    y_min = min(y for points in series.values() for _, y in points)
    y_max = max(y for points in series.values() for _, y in points)
    if y_min <= 0 <= y_max:
        ax.axhline(y=0, color=MUTED, linestyle="--", linewidth=1.5, alpha=0.8)

    # Add legend
    ax.legend(loc="best", fontsize=10, frameon=True, fancybox=True, shadow=False)

    # Add title and subtitle
    fig.suptitle(title, fontsize=16, fontweight="bold", color=INK, x=0.08, y=0.96, ha="left")
    ax.text(0.08, 0.91, subtitle, transform=fig.transFigure, fontsize=11, color=MUTED, ha="left")

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    return path
