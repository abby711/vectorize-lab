"""The first version of the tracer, kept for comparison.

Orchestrates the full raster -> SVG pipeline.

    decode -> quantize -> label regions -> extract rings
           -> simplify -> fit curves -> emit SVG
"""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from ..core.colour import quantize_full_color, quantize_grayscale, quantize_monochrome
from ..core.result import VectorizeRejected, VectorizeResult
from . import curves, polyline, regions, svg

MODE_FULL_COLOR = "full_color"
MODE_GRAYSCALE = "grayscale"
MODE_MONOCHROME = "monochrome"


@dataclass
class VectorizeOptions:
    mode: str = MODE_FULL_COLOR
    color_count: int = 8
    gray_count: int = 5
    threshold: int = 128
    max_dimension: int = 640
    upscale: float = 1.0
    presmooth: int = 0
    simplify_epsilon: float = 1.6
    corner_threshold_degrees: float = 42.0
    corner_window: int = 2
    smoothness: float = 1.0
    min_region_area: int = 12
    max_regions: int = 4000
    fit_curves: bool = True
    drop_largest_region: bool = False


def _load_rgba(path_or_file, max_dimension, upscale, presmooth=0):
    """Returns (rgba, scale) where scale maps source pixels -> traced pixels.

    Upscaling before tracing thickens thin strokes in pixel terms, which
    keeps them from being eaten by simplification.
    """
    image = Image.open(path_or_file).convert("RGBA")
    scale = 1.0

    if upscale and upscale > 1.0:
        scale = float(upscale)
        target = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(target, Image.LANCZOS)

    longest = max(image.size)
    if max_dimension and longest > max_dimension:
        shrink = max_dimension / float(longest)
        scale *= shrink
        target = (max(1, round(image.width * shrink)), max(1, round(image.height * shrink)))
        image = image.resize(target, Image.LANCZOS)

    # Edge-preserving denoise before quantisation. Sensor noise and JPEG
    # ringing otherwise survive as real region boundaries, which is what
    # shatters a photograph into hundreds of speckles.
    if presmooth and presmooth >= 3:
        size = presmooth if presmooth % 2 == 1 else presmooth + 1
        alpha = image.getchannel("A")
        smoothed = image.convert("RGB").filter(ImageFilter.MedianFilter(size))
        image = smoothed.convert("RGBA")
        image.putalpha(alpha)

    return np.array(image), scale


def _quantize(rgb, options):
    if options.mode == MODE_MONOCHROME:
        return quantize_monochrome(rgb, options.threshold)
    if options.mode == MODE_GRAYSCALE:
        return quantize_grayscale(rgb, options.gray_count)
    return quantize_full_color(rgb, options.color_count)


def _collect_regions(label_map, opaque, palette_size, options):
    """Segments first, counts, and rejects before any curve work happens.

    Gating here rather than after tracing is what keeps a noise-heavy
    image from costing minutes to produce something unusable.
    """
    qualifying = []

    for palette_index in range(palette_size):
        mask = (label_map == palette_index) & opaque
        if not mask.any():
            continue

        component_map, component_count = regions.label_components(mask)
        for component in range(component_count):
            area = int((component_map == component).sum())
            if area < options.min_region_area:
                continue
            qualifying.append((area, palette_index, component_map, component))

            if options.max_regions and len(qualifying) > options.max_regions:
                raise VectorizeRejected(
                    f"input produced more than {options.max_regions} regions — "
                    "too detailed or too noisy for region tracing",
                    len(qualifying),
                )

    return qualifying


def vectorize_image(path_or_file, options=None):
    options = options or VectorizeOptions()
    rgba, scale = _load_rgba(
        path_or_file, options.max_dimension, options.upscale, options.presmooth
    )
    height, width = rgba.shape[:2]

    rgb = rgba[..., :3]
    opaque = rgba[..., 3] >= 128

    label_map, palette = _quantize(rgb, options)
    corner_threshold = math.radians(options.corner_threshold_degrees)
    inverse_scale = 1.0 / scale if scale else 1.0

    qualifying = _collect_regions(label_map, opaque, len(palette), options)

    candidates = []
    for area, palette_index, component_map, component in qualifying:
        component_mask = component_map == component

        rings = regions.extract_rings(component_mask)
        if not rings:
            continue

        rings_segments = []
        segment_count = 0
        for ring in rings:
            reduced = polyline.simplify_ring(ring, options.simplify_epsilon)
            if len(reduced) < 3:
                continue
            if options.fit_curves:
                segments = curves.fit_ring(
                    reduced, corner_threshold, options.smoothness, options.corner_window
                )
            else:
                segments = curves.polygon_segments(reduced)
            if segments:
                rings_segments.append((reduced[0], segments))
                segment_count += len(segments)

        if rings_segments:
            candidates.append((area, palette_index, rings_segments, segment_count))

    # Paint largest first so small details land on top.
    candidates.sort(key=lambda item: item[0], reverse=True)
    if options.drop_largest_region and candidates:
        candidates = candidates[1:]

    shapes = []
    total_segments = 0
    for _, palette_index, rings_segments, segment_count in candidates:
        shapes.append(
            {
                "data": svg.path_data(rings_segments, inverse_scale),
                "fill": svg.to_hex(palette[palette_index]),
            }
        )
        total_segments += segment_count

    output_width = int(round(width * inverse_scale))
    output_height = int(round(height * inverse_scale))

    return VectorizeResult(
        svg=svg.build_svg(shapes, output_width, output_height),
        width=output_width,
        height=output_height,
        shape_count=len(shapes),
        path_segment_count=total_segments,
        palette=[svg.to_hex(entry) for entry in palette],
    )
