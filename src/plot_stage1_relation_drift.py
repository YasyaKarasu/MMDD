#!/usr/bin/env python
"""Plot the shared-model relation-drift trade-off from a Stage-1 history."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


BLUE = "#1F5A94"
ORANGE = "#D97706"
INK = "#20262E"
MUTED = "#667085"
GRID = "#DDE2E8"
BACKGROUND = "#FFFFFF"


def load_points(history_path: Path) -> list[dict[str, float | int]]:
    history = json.loads(history_path.read_text(encoding="utf-8"))
    points = []
    for record in history["epochs"]:
        by_dataset = record["dev_retrieval"]["by_dataset"]
        points.append(
            {
                "epoch": int(record["epoch"]),
                "drift": float(record["relation_drift"]["table_to_table"]),
                "entitables": float(
                    by_dataset["entitables20k_v4"]["direct"]["recall@10"]
                ),
                "wdc": float(by_dataset["wdc2k_v2"]["direct"]["recall@10"]),
            }
        )
    if len(points) < 2:
        raise ValueError("relation-drift plot requires at least two epochs")
    return sorted(points, key=lambda point: int(point["epoch"]))


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size=size)


def _dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: str,
    width: int,
    dash: float = 10,
    gap: float = 7,
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    offset = 0.0
    while offset < length:
        segment_end = min(offset + dash, length)
        draw.line(
            (
                start[0] + dx * offset / length,
                start[1] + dy * offset / length,
                start[0] + dx * segment_end / length,
                start[1] + dy * segment_end / length,
            ),
            fill=fill,
            width=width,
        )
        offset += dash + gap


def _panel(
    draw: ImageDraw.ImageDraw,
    points: list[dict[str, float | int]],
    *,
    bounds: tuple[int, int, int, int],
    key: str,
    label: str,
    color: str,
    marker: str,
    dashed: bool,
    x_max: float,
) -> None:
    left, top, right, bottom = bounds
    values = [float(point[key]) * 100 for point in points]
    y_min = math.floor(min(values)) - 1
    y_max = math.ceil(max(values)) + 1
    if y_max == y_min:
        y_max += 1

    title_font = _font(28, bold=True)
    tick_font = _font(18)
    annotation_font = _font(17, bold=True)
    draw.text((left, top - 48), label, font=title_font, fill=INK)

    plot_top = top
    plot_bottom = bottom
    plot_left = left + 58
    plot_right = right - 20
    for tick_index in range(5):
        fraction = tick_index / 4
        y = plot_bottom - fraction * (plot_bottom - plot_top)
        value = y_min + fraction * (y_max - y_min)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
        tick = f"{value:.1f}%"
        tick_box = draw.textbbox((0, 0), tick, font=tick_font)
        draw.text(
            (plot_left - (tick_box[2] - tick_box[0]) - 10, y - 10),
            tick,
            font=tick_font,
            fill=MUTED,
        )
    for tick in range(6):
        x = plot_left + tick / 5 * (plot_right - plot_left)
        draw.line((x, plot_bottom, x, plot_bottom + 6), fill=INK, width=2)
        label_text = str(tick)
        label_box = draw.textbbox((0, 0), label_text, font=tick_font)
        draw.text(
            (x - (label_box[2] - label_box[0]) / 2, plot_bottom + 12),
            label_text,
            font=tick_font,
            fill=MUTED,
        )
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=INK, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=INK, width=2)

    def xy(point: dict[str, float | int]) -> tuple[float, float]:
        x = plot_left + float(point["drift"]) / x_max * (plot_right - plot_left)
        value = float(point[key]) * 100
        y = plot_bottom - (value - y_min) / (y_max - y_min) * (
            plot_bottom - plot_top
        )
        return x, y

    coordinates = [xy(point) for point in points]
    for start, end in zip(coordinates, coordinates[1:]):
        if dashed:
            _dashed_line(draw, start, end, fill=color, width=4)
        else:
            draw.line((*start, *end), fill=color, width=4)
    for coordinate in coordinates:
        x, y = coordinate
        if marker == "circle":
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline=INK)
        else:
            draw.rectangle(
                (x - 6, y - 6, x + 6, y + 6), fill=BACKGROUND, outline=color, width=4
            )

    for point_index in (0, min(8, len(points) - 1), len(points) - 1):
        point = points[point_index]
        x, y = coordinates[point_index]
        text = f"e{point['epoch']}  {float(point[key]):.1%}"
        text_box = draw.textbbox((0, 0), text, font=annotation_font)
        text_width = text_box[2] - text_box[0]
        text_x = min(max(x - text_width / 2, plot_left), plot_right - text_width)
        text_y = y - 30 if point_index != 0 else y + 12
        draw.text((text_x, text_y), text, font=annotation_font, fill=color)

    axis_label = "||R(table→table) − I||F"
    axis_box = draw.textbbox((0, 0), axis_label, font=tick_font)
    draw.text(
        (
            (plot_left + plot_right - (axis_box[2] - axis_box[0])) / 2,
            plot_bottom + 48,
        ),
        axis_label,
        font=tick_font,
        fill=INK,
    )


def render_chart(points: list[dict[str, float | int]], output_path: Path) -> None:
    width, height = 1400, 720
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text(
        (62, 28),
        "Shared-model relation drift vs per-lake direct R@10",
        font=_font(36, bold=True),
        fill=INK,
    )
    draw.text(
        (62, 78),
        "r3 Task H, ensemble-KD weight 0.3 · 11 epochs · panel-specific y scales",
        font=_font(21),
        fill=MUTED,
    )
    x_max = max(5.0, math.ceil(max(float(point["drift"]) for point in points)))
    _panel(
        draw,
        points,
        bounds=(60, 170, 690, 600),
        key="entitables",
        label="EntiTables (n=938)",
        color=BLUE,
        marker="circle",
        dashed=False,
        x_max=x_max,
    )
    _panel(
        draw,
        points,
        bounds=(740, 170, 1370, 600),
        key="wdc",
        label="WDC (n=202)",
        color=ORANGE,
        marker="square",
        dashed=True,
        x_max=x_max,
    )
    draw.text(
        (62, 682),
        "Source: stage1_optimization_r3_20260829/taskH_ensemble_edge_kd_w0.3 history. "
        "Markers follow epoch order; e0, e8, and final epoch are labeled.",
        font=_font(17),
        fill=MUTED,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render_chart(load_points(args.history), args.output)
