#!/usr/bin/env python3
"""CLI front end for vectorlab.

    ./vectorize.py in.png -o out.svg                    # v2, no tuning needed
    ./vectorize.py in.png -o out.svg --tolerance 1.0    # fewer anchors, slightly looser
    ./vectorize.py in.png -o out.svg --engine v1 --colors 8
"""

import argparse
import sys

from vectorlab import VectorizeOptions, VectorizeRejected, vectorize_image
from vectorlab.pipeline import MODE_FULL_COLOR, MODE_GRAYSCALE, MODE_MONOCHROME


def build_parser():
    parser = argparse.ArgumentParser(description="Raster to SVG tracer")
    parser.add_argument("input", help="PNG / JPG / WEBP input")
    parser.add_argument("-o", "--output", required=True, help="SVG output path")
    parser.add_argument("--engine", default="v2", choices=["v2", "v1"], help="tracing engine")

    v2 = parser.add_argument_group("v2")
    v2.add_argument("--tolerance", type=float, default=0.6, help="max curve error in pixels")
    v2.add_argument("--max-colors", type=int, default=32, help="cap on the detected palette")
    v2.add_argument("--no-primitives", action="store_true", help="always emit paths")
    v2.add_argument(
        "--max-regions", type=int, default=6000, help="region budget; above it, small regions merge harder"
    )

    v1 = parser.add_argument_group("v1")
    v1.add_argument(
        "--mode",
        default=MODE_FULL_COLOR,
        choices=[MODE_FULL_COLOR, MODE_GRAYSCALE, MODE_MONOCHROME],
        help="colour mode",
    )
    v1.add_argument("--colors", type=int, default=8, help="palette size for full_color")
    v1.add_argument("--grays", type=int, default=5, help="band count for grayscale")
    v1.add_argument("--threshold", type=int, default=128, help="0-255 cut for monochrome")
    v1.add_argument("--max-dim", type=int, default=640, help="downscale longest edge")
    v1.add_argument("--upscale", type=float, default=1.0, help="resample up before tracing")
    v1.add_argument("--presmooth", type=int, default=0, help="median filter radius before quantizing")
    v1.add_argument("--epsilon", type=float, default=1.6, help="simplification, in pixels")
    v1.add_argument("--corner", type=float, default=42.0, help="corner angle, in degrees")
    v1.add_argument("--corner-window", type=int, default=2, help="vertices each side for corner angle")
    v1.add_argument("--smoothness", type=float, default=1.0, help="tangent scale, 0 = polygons")
    v1.add_argument("--min-area", type=int, default=12, help="drop regions below this area")
    v1.add_argument("--no-fit", action="store_true", help="emit polygons, skip curve fitting")
    v1.add_argument("--drop-background", action="store_true", help="omit the largest region")
    return parser


def _run_v1(args):
    options = VectorizeOptions(
        mode=args.mode,
        color_count=args.colors,
        gray_count=args.grays,
        threshold=args.threshold,
        max_dimension=args.max_dim,
        upscale=args.upscale,
        presmooth=args.presmooth,
        simplify_epsilon=args.epsilon,
        corner_threshold_degrees=args.corner,
        corner_window=args.corner_window,
        smoothness=args.smoothness,
        min_region_area=args.min_area,
        fit_curves=not args.no_fit,
        drop_largest_region=args.drop_background,
    )
    return vectorize_image(args.input, options)


def _run_v2(args):
    from vectorlab.v2 import V2Options, vectorize_v2

    options = V2Options(
        tolerance=args.tolerance,
        max_colors=args.max_colors,
        use_primitives=not args.no_primitives,
        max_regions=args.max_regions,
    )
    return vectorize_v2(args.input, options)


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        result = _run_v2(args) if args.engine == "v2" else _run_v1(args)
    except VectorizeRejected as rejected:
        print(f"rejected: {rejected.reason}")
        return 2

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(result.svg)

    print(
        f"{args.output}: {result.width}x{result.height}, "
        f"{result.shape_count} shapes, {result.path_segment_count} anchors, "
        f"palette {' '.join(result.palette)}"
    )
    for note in getattr(result, "notes", ()):
        print(f"note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
