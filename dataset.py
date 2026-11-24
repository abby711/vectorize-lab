#!/usr/bin/env python3
"""Builds raster fixtures from SVG originals, with ground truth.

The point: when the source is an SVG, you know what a human or a vector
tool actually produced. Rasterise it, trace it back, and you can ask the
question that matters — how close is our output to the real thing —
instead of only comparing pixels against the bitmap we were handed.

    python3 dataset.py                   # dataset/svg/*.svg -> samples/bench/
    python3 dataset.py --width 512

Writes `samples/bench/ground-truth.json` recording each original's node
and path counts, which `compare.py` reports against.
"""

import argparse
import json
import os

import re

from vectorlab.core.metrics import count_editable_points

IMAGE_WIDTH = 480


def raster_width(svg_text, fallback):
    """data-raster-width wins, then the SVG's own width, then the fallback."""
    root = re.search(r"<svg\b[^>]*>", svg_text)
    if root:
        for attribute in ("data-raster-width", "width"):
            found = re.search(rf'\b{attribute}="([\d.]+)"', root.group(0))
            if found:
                return int(float(found.group(1)))
    return fallback


def is_transparent(svg_text):
    """data-transparent on the root keeps the raster's background clear."""
    root = re.search(r"<svg\b[^>]*>", svg_text)
    return bool(root and re.search(r'\bdata-transparent="(true|1)"', root.group(0)))


def rasterize(svg_path, png_path, width, transparent=False):
    import cairosvg

    cairosvg.svg2png(
        url=svg_path,
        write_to=png_path,
        output_width=width,
        background_color=None if transparent else "white",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="SVG originals -> raster fixtures")
    parser.add_argument("--svg-dir", default="dataset/svg", help="folder of SVG originals")
    parser.add_argument("--out", default="samples/bench", help="where PNGs are written")
    parser.add_argument(
        "--width", type=int, default=0, help="force one raster width; default is each SVG's own"
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.svg_dir):
        parser.error(f"no such folder: {args.svg_dir}")

    names = sorted(name for name in os.listdir(args.svg_dir) if name.lower().endswith(".svg"))
    if not names:
        parser.error(f"no .svg files in {args.svg_dir}")

    os.makedirs(args.out, exist_ok=True)
    truth_path = os.path.join(args.out, "ground-truth.json")

    truth = {}

    for name in names:
        svg_path = os.path.join(args.svg_dir, name)
        stem = os.path.splitext(name)[0]
        png_name = f"{stem}.png"
        png_path = os.path.join(args.out, png_name)

        with open(svg_path, encoding="utf-8") as handle:
            text = handle.read()
        geometry = count_editable_points(text)
        width = args.width or raster_width(text, IMAGE_WIDTH)

        transparent = is_transparent(text)
        rasterize(svg_path, png_path, width, transparent)
        truth[png_name] = {"source": name, "width": width, "transparent": transparent, **geometry}

        print(
            f"{png_name}: {width}px wide, {geometry['shapes']} shapes, "
            f"{geometry['anchors']} anchors in the original"
        )

    with open(truth_path, "w", encoding="utf-8") as handle:
        json.dump(truth, handle, indent=2, sort_keys=True)

    print(f"\nwrote {truth_path} ({len(truth)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
