#!/usr/bin/env python3
"""Property tests: invariants checked over random scenes, not just fixtures.

Each scene is a seeded random SVG, so any failure reproduces exactly.
Runs standalone (`python3 tests/test_properties.py`) or under pytest.
"""

import io
import json
import math
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cairosvg  # noqa: E402
from PIL import Image  # noqa: E402

from vectorlab.metrics import count_editable_points  # noqa: E402
from vectorlab.pipeline import VectorizeRejected  # noqa: E402
from vectorlab.v2 import V2Options, vectorize_v2  # noqa: E402
from vectorlab.v2.contour import coverage_field, trace_loops  # noqa: E402
from vectorlab.v2.partition import build_partition  # noqa: E402
from vectorlab.v2.primitives import detect_primitive  # noqa: E402

SEEDS = range(12)
WIDTH, HEIGHT = 240, 180


def _colour(rng):
    return "#{:02x}{:02x}{:02x}".format(*(rng.randrange(256) for _ in range(3)))


def random_scene(seed):
    """A flat illustration: background plus 3–7 overlapping primitives and blobs."""
    rng = random.Random(seed)
    parts = [f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{_colour(rng)}"/>']
    for _ in range(rng.randint(3, 7)):
        kind = rng.choice(["circle", "ellipse", "rect", "polygon", "blob"])
        cx, cy = rng.uniform(30, WIDTH - 30), rng.uniform(30, HEIGHT - 30)
        size = rng.uniform(12, 55)
        fill = _colour(rng)
        if kind == "circle":
            parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{size:.2f}" fill="{fill}"/>')
        elif kind == "ellipse":
            parts.append(
                f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{size:.2f}" ry="{size * rng.uniform(0.4, 0.9):.2f}" '
                f'transform="rotate({rng.uniform(0, 180):.1f} {cx:.2f} {cy:.2f})" fill="{fill}"/>'
            )
        elif kind == "rect":
            width, height = size * 1.6, size * rng.uniform(0.5, 1.2)
            parts.append(
                f'<rect x="{cx - width / 2:.2f}" y="{cy - height / 2:.2f}" width="{width:.2f}" '
                f'height="{height:.2f}" rx="{rng.choice([0, size * 0.2]):.2f}" fill="{fill}"/>'
            )
        elif kind == "polygon":
            points = " ".join(
                f"{cx + size * math.cos(2 * math.pi * index / 5 + seed):.2f},"
                f"{cy + size * math.sin(2 * math.pi * index / 5 + seed):.2f}"
                for index in range(rng.randint(3, 6))
            )
            parts.append(f'<polygon points="{points}" fill="{fill}"/>')
        else:
            radii = [size * rng.uniform(0.6, 1.0) for _ in range(4)]
            anchors = [
                (cx + radii[index] * math.cos(index * math.pi / 2), cy + radii[index] * math.sin(index * math.pi / 2))
                for index in range(4)
            ]
            path = f"M{anchors[0][0]:.2f} {anchors[0][1]:.2f}"
            for index in range(4):
                start, end = anchors[index], anchors[(index + 1) % 4]
                handle = 0.55 * size
                c1 = (start[0] - handle * math.sin(index * math.pi / 2), start[1] + handle * math.cos(index * math.pi / 2))
                c2 = (end[0] + handle * math.sin((index + 1) * math.pi / 2), end[1] - handle * math.cos((index + 1) * math.pi / 2))
                path += f" C{c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} {end[0]:.2f} {end[1]:.2f}"
            parts.append(f'<path d="{path} Z" fill="{fill}"/>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}">' + "".join(parts) + "</svg>"
    )


def _rasterise(svg_text, background=None):
    payload = cairosvg.svg2png(
        bytestring=svg_text.encode("utf-8"),
        output_width=WIDTH,
        output_height=HEIGHT,
        background_color=background,
    )
    return np.array(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.int16)


def _trace(seed, options=None):
    source = _rasterise(random_scene(seed), "white").astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(source).save(buffer, format="PNG")
    buffer.seek(0)
    return source, vectorize_v2(buffer, options)


# ------------------------------------------------------------------ properties


def test_every_pixel_painted():
    """Render over two different backgrounds: any pixel that changes was left uncovered."""
    for seed in SEEDS:
        _, result = _trace(seed)
        over_magenta = _rasterise(result.svg, "#ff00ff")
        over_green = _rasterise(result.svg, "#00ff00")
        exposed = np.abs(over_magenta - over_green).max(axis=2) > 24
        fraction = float(exposed.mean())
        assert fraction < 0.0005, f"seed {seed}: {fraction:.4%} of pixels show the background"


def test_faithful():
    """Mean colour error stays small on every random scene."""
    for seed in SEEDS:
        source, result = _trace(seed)
        rendered = _rasterise(result.svg, "white")
        error = float(np.abs(rendered - source.astype(np.int16)).mean())
        assert error < 1.0, f"seed {seed}: mean error {error:.2f}"


def test_deterministic():
    for seed in SEEDS[:4]:
        _, first = _trace(seed)
        _, second = _trace(seed)
        assert first.svg == second.svg, f"seed {seed}: output differs between runs"


def test_closed_and_renderable():
    for seed in SEEDS:
        _, result = _trace(seed)
        for data in __import__("re").findall(r'\bd="([^"]*)"', result.svg):
            assert data.count("M") == data.count("Z"), f"seed {seed}: unclosed subpath"
        _rasterise(result.svg)


def test_partition_never_drops_pixels():
    """Every pixel belongs to exactly one region, and no region is undersized."""
    for seed in SEEDS:
        source = _rasterise(random_scene(seed), "white").astype(np.uint8)
        options = V2Options()
        from vectorlab.v2 import _PartitionView

        partition = build_partition(source, _PartitionView(options, options.blend_residual))
        assert partition.component_map.min() >= 0
        assert int(partition.areas.sum()) == source.shape[0] * source.shape[1]
        if len(partition.areas) > 1:
            assert int(partition.areas.min()) >= options.min_region_area, f"seed {seed}"


def test_subpixel_contour_accuracy():
    """A traced circle sits within a tenth of a pixel of the true one, on average."""
    rng = random.Random(7)
    for _ in range(6):
        cx, cy, radius = rng.uniform(80, 160), rng.uniform(70, 110), rng.uniform(20, 60)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}">'
            f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="#1d3557"/></svg>'
        )
        rgb = _rasterise(svg, "white").astype(np.uint8)
        palette = np.array([[255, 255, 255], [0x1D, 0x35, 0x57]], dtype=np.float64) / 255.0
        labels = np.argmin(
            np.linalg.norm(rgb[..., None, :] / 255.0 - palette[None, None], axis=-1), axis=-1
        ).astype(np.int32)
        field = coverage_field(labels == 1, labels, rgb / 255.0, palette)
        loop = trace_loops(np.pad(field, 1))[0] - 0.5
        error = np.abs(np.hypot(loop[:, 0] - cx, loop[:, 1] - cy) - radius)
        assert error.mean() < 0.1, f"mean radial error {error.mean():.3f}px"


def test_primitive_recovery():
    """Recognised primitives come back with their true parameters."""
    rng = random.Random(11)
    for _ in range(6):
        cx, cy, radius = rng.uniform(80, 160), rng.uniform(70, 110), rng.uniform(15, 50)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}">'
            f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="#e63946"/></svg>'
        )
        rgb = _rasterise(svg, "white").astype(np.uint8)
        palette = np.array([[255, 255, 255], [0xE6, 0x39, 0x46]], dtype=np.float64) / 255.0
        labels = np.argmin(
            np.linalg.norm(rgb[..., None, :] / 255.0 - palette[None, None], axis=-1), axis=-1
        ).astype(np.int32)
        loop = trace_loops(np.pad(coverage_field(labels == 1, labels, rgb / 255.0, palette), 1))[0] - 0.5
        found = detect_primitive(loop, 0.6)
        assert found is not None and found.kind == "circle"
        assert abs(found.centre[0] - cx) < 0.3 and abs(found.centre[1] - cy) < 0.3
        assert abs(found.size[0] - radius) < 0.3


def test_rejects_rather_than_grinds():
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(noise).save(buffer, format="PNG")
    buffer.seek(0)
    try:
        vectorize_v2(buffer, V2Options(max_regions=200))
    except VectorizeRejected:
        return
    raise AssertionError("pure noise was traced instead of rejected")


JPEG_GUARD = 5.0  # today's level; the target is 2.0 — see docs/APPROACH.md, known gaps


def test_jpeg_close_to_png():
    """Compression shouldn't change what was drawn: a JPEG copy traces near its PNG.

    Regression guard, not a goal. Small JPEGs (240 px here) currently trace
    at 1.1-4.7x their PNG's anchors; tighten JPEG_GUARD as that improves.
    """
    for seed in SEEDS[:5]:
        source = _rasterise(random_scene(seed), "white").astype(np.uint8)
        anchors = {}
        for fmt in ("PNG", "JPEG"):
            buffer = io.BytesIO()
            Image.fromarray(source).save(buffer, format=fmt, quality=82)
            buffer.seek(0)
            anchors[fmt] = vectorize_v2(buffer).path_segment_count
        assert anchors["JPEG"] <= JPEG_GUARD * anchors["PNG"] + 10, f"seed {seed}: {anchors}"


KNOWN_LIMITS = {
    "23-gradient-sky.png": "gradients are traced as bands, not gradient fills",
    "24-line-art.png": "strokes are traced as filled outlines, not centrelines",
}


def test_ground_truth_regression():
    """Guard rails on the designer-drawn fixtures, minus documented limitations."""
    folder = os.path.join(ROOT, "samples", "bench")
    with open(os.path.join(folder, "ground-truth.json"), encoding="utf-8") as handle:
        truth = json.load(handle)
    for name, meta in truth.items():
        if name in KNOWN_LIMITS:
            continue
        path = os.path.join(folder, name)
        result = vectorize_v2(path)
        anchors = count_editable_points(result.svg)["anchors"]
        assert anchors <= 1.8 * meta["anchors"], f"{name}: {anchors} anchors vs {meta['anchors']}"


# ------------------------------------------------------------------ runner

if __name__ == "__main__":
    failures = 0
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    for test in tests:
        try:
            test()
            print(f"pass  {test.__name__}")
        except AssertionError as error:
            failures += 1
            print(f"FAIL  {test.__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
