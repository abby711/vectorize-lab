#!/usr/bin/env python3
"""Rasterises the emitted SVG and scores it against the source image.

No SVG renderer needed: the path grammar we emit is a closed subset
(M / L / C / Z), so we flatten the Beziers ourselves and fill with PIL.
Reported score is mean per-pixel agreement plus a colour delta.
"""

import re
import sys

import numpy as np
from PIL import Image, ImageDraw

TOKEN = re.compile(r"([MLCZ])([^MLCZ]*)")


def _numbers(text):
    return [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _flatten_cubic(p0, p1, p2, p3, steps=12):
    t = np.linspace(0.0, 1.0, steps + 1)[1:]
    one = 1.0 - t
    points = (
        np.outer(one**3, p0)
        + np.outer(3 * one**2 * t, p1)
        + np.outer(3 * one * t**2, p2)
        + np.outer(t**3, p3)
    )
    return [tuple(point) for point in points]


def parse_path(data):
    """-> list of rings, each a list of (x, y)."""
    rings = []
    current = []
    cursor = (0.0, 0.0)

    for command, payload in TOKEN.findall(data):
        values = _numbers(payload)
        if command == "M":
            if len(current) >= 3:
                rings.append(current)
            cursor = (values[0], values[1])
            current = [cursor]
        elif command == "L":
            for index in range(0, len(values), 2):
                cursor = (values[index], values[index + 1])
                current.append(cursor)
        elif command == "C":
            for index in range(0, len(values), 6):
                control_1 = np.array(values[index : index + 2])
                control_2 = np.array(values[index + 2 : index + 4])
                end = np.array(values[index + 4 : index + 6])
                current.extend(_flatten_cubic(np.array(cursor), control_1, control_2, end))
                cursor = tuple(end)
        elif command == "Z":
            if len(current) >= 3:
                rings.append(current)
            current = []

    if len(current) >= 3:
        rings.append(current)
    return rings


def render_svg(svg_text, size):
    width, height = size
    canvas = Image.new("RGB", (width, height), (255, 255, 255))

    for element in re.finditer(r"<path\b[^>]*/?>", svg_text):
        markup = element.group(0)

        data_match = re.search(r'\bd="([^"]+)"', markup)
        fill_match = re.search(r'\bfill="(#[0-9a-fA-F]{6})"', markup)
        if not data_match or not fill_match:
            continue

        fill = fill_match.group(1)
        colour = tuple(int(fill[i : i + 2], 16) for i in (1, 3, 5))
        rings = parse_path(data_match.group(1))
        if not rings:
            continue

        # Other tracers place each path with its own translate, so the
        # path data alone is not in document coordinates.
        shift = re.search(r"translate\(\s*(-?[\d.]+)[ ,]+(-?[\d.]+)\s*\)", markup)
        if shift:
            offset_x = float(shift.group(1))
            offset_y = float(shift.group(2))
            rings = [
                [(point[0] + offset_x, point[1] + offset_y) for point in ring] for ring in rings
            ]

        # Even-odd across all rings of this shape: XOR successive fills.
        layer = Image.new("1", (width, height), 0)
        for ring in rings:
            single = Image.new("1", (width, height), 0)
            ImageDraw.Draw(single).polygon(ring, fill=1)
            layer = Image.fromarray(
                (np.array(layer) ^ np.array(single)).astype(np.uint8) * 255
            ).convert("1")

        canvas.paste(Image.new("RGB", (width, height), colour), mask=layer)

    return canvas


def render_with_cairo(svg_path, size):
    """Render through a real SVG engine, so any tracer's output is fair game.

    The built-in parser only understands the closed M/L/C/Z subset this
    project emits. Other engines use relative commands, shorthands and
    arcs, and silently mis-score under it.
    """
    import io

    import cairosvg

    width, height = size
    payload = cairosvg.svg2png(
        url=svg_path, output_width=width, output_height=height, background_color="white"
    )
    return Image.open(io.BytesIO(payload)).convert("RGB")


def score(source_path, svg_path, renderer="auto"):
    """renderer: 'auto' prefers cairosvg, 'builtin' forces the local parser."""
    source = Image.open(source_path).convert("RGB")

    rendered = None
    if renderer != "builtin":
        try:
            rendered = render_with_cairo(svg_path, source.size)
        except Exception:  # noqa: BLE001 - fall back rather than fail the run
            rendered = None

    if rendered is None:
        with open(svg_path, encoding="utf-8") as handle:
            rendered = render_svg(handle.read(), source.size)
    left = np.array(source, dtype=np.int16)
    right = np.array(rendered, dtype=np.int16)

    delta = np.abs(left - right).mean(axis=2)
    close = float((delta < 32).mean())
    return close, float(delta.mean()), rendered


if __name__ == "__main__":
    source_path, svg_path = sys.argv[1], sys.argv[2]
    agreement, mean_delta, rendered = score(source_path, svg_path)
    if len(sys.argv) > 3:
        rendered.save(sys.argv[3])
    print(
        f"{svg_path}: {agreement * 100:.1f}% pixels within tolerance, "
        f"mean channel delta {mean_delta:.1f}"
    )
