"""vectorlab v2 — sub-pixel, globally fitted, primitive-aware tracing.

    decode → detect palette → blend-aware labels → partition (merge, never drop)
           → paint order by enclosed area, holes filled
           → coverage field → marching squares → primitive or optimal spline
           → SVG

Every idea is a switch on V2Options so each one's contribution can be
measured on its own.
"""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from ..metrics import count_editable_points
from ..pipeline import VectorizeRejected, VectorizeResult
from .contour import coverage_field, trace_loops
from .emit import build_document, path_data, primitive_element, to_hex
from .fitting import FitSettings, fit_loops
from .partition import label_image, label_partition, partition_labels
from .primitives import detect_primitive

_MARGIN = 2


@dataclass
class V2Options:
    # palette and partition
    max_colors: int = 32
    min_palette_fraction: float = 0.0004
    palette_merge_delta: float = 7.0
    pure_delta: float = 5.0
    min_flat_coverage: float = 0.55
    blend_residual: float = 0.07
    fallback_colors: int = 12
    max_fallback_colors: int = 96
    target_delta: float = 4.0
    min_region_area: int = 4
    noisy_region_area: int = 16
    keep_contrast: float = 20.0
    residual_delta: float = 12.0
    jpeg_blend_reach: int = 3
    min_missing_colour: int = 24
    max_regions: int = 6000
    max_noise: float = 12.0
    max_dimension: int = 1024

    # fitting
    tolerance: float = 0.6
    fit: FitSettings = field(default_factory=FitSettings)

    # the ideas, each switchable for ablation
    use_coverage: bool = True
    use_optimal_fit: bool = True
    use_primitives: bool = True
    use_stacking: bool = True
    use_occlusion: bool = True
    use_blend_labels: bool = True


def _load(path_or_file, max_dimension):
    """Returns (rgb, scale, was_jpeg)."""
    opened = Image.open(path_or_file)
    was_jpeg = opened.format == "JPEG"
    image = opened.convert("RGB")
    scale = 1.0
    longest = max(image.size)
    if max_dimension and longest > max_dimension:
        scale = max_dimension / float(longest)
        # Box filtering, not Lanczos: coverage recovery assumes an edge
        # pixel is an area average, and Lanczos ringing paints false halos.
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.BOX
        )
    return np.array(image), scale, was_jpeg


def _fill_holes(mask):
    """Mask plus every enclosed pocket. Background is 8-connected, matching the tracer."""
    outside, values = label_partition((~mask).astype(np.int8), connectivity=8)
    border = np.unique(
        np.concatenate((outside[0], outside[-1], outside[:, 0], outside[:, -1]))
    )
    pocket = ~mask & ~np.isin(outside, border)
    return mask | pocket


_SUPERSAMPLE = 4


def _hull_polygon(mask):
    """Convex hull of a mask's pixel squares, as (x, y) corners in crop pixel units.

    Only each row's outermost pixels can be hull vertices, so the hull is
    built from those rather than from every pixel.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    if not len(rows):
        return []
    left = np.argmax(mask[rows], axis=1)
    right = mask.shape[1] - 1 - np.argmax(mask[rows, ::-1], axis=1)
    corners = np.concatenate(
        [
            np.stack([left, rows], 1),
            np.stack([left, rows + 1], 1),
            np.stack([right + 1, rows], 1),
            np.stack([right + 1, rows + 1], 1),
        ]
    ).astype(np.float64)
    ordered = corners[np.lexsort((corners[:, 1], corners[:, 0]))]

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])

    lower, upper = [], []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in ordered[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return [tuple(point) for point in lower[:-1] + upper[:-1]]


def _polygon_coverage(polygon, shape):
    """Antialiased coverage of a polygon over a crop, by supersampling."""
    height, width = shape
    canvas = Image.new("L", (width * _SUPERSAMPLE, height * _SUPERSAMPLE), 0)
    ImageDraw.Draw(canvas).polygon(
        [(x * _SUPERSAMPLE - 0.5, y * _SUPERSAMPLE - 0.5) for x, y in polygon], fill=255
    )
    fine = np.array(canvas, dtype=np.float64) / 255.0
    return fine.reshape(height, _SUPERSAMPLE, width, _SUPERSAMPLE).mean(axis=(1, 3))


def _occluded_extension(filled, component_crop, rank, own_rank):
    """Grow a shape under the shapes painted on top of it, within its convex hull.

    A designer draws the sky as one rectangle and paints mountains over
    it; tracing only the visible sky gives an outline that follows every
    ridge. Anything added here is overpainted later, so the picture is
    unchanged while the outline gets simpler.

    Returns (grown mask, hull coverage or None).
    """
    later = np.zeros(filled.shape, dtype=bool)
    valid = component_crop >= 0
    later[valid] = rank[component_crop[valid]] > own_rank
    polygon = _hull_polygon(filled)
    if len(polygon) < 3 or not later.any():
        return filled, None

    hull_coverage = _polygon_coverage(polygon, filled.shape)
    grown = filled | ((hull_coverage >= 0.5) & later)
    if grown.sum() == filled.sum():
        return filled, None

    pieces, _ = label_partition(grown.astype(np.int8))
    seeds = np.unique(pieces[filled])
    return np.isin(pieces, seeds) & grown, hull_coverage


def _blend_hidden_edges(coverage, measured, mask, hull_coverage):
    """Use hull coverage on boundary pixels that had no colour edge to unmix."""
    padded = np.pad(mask, 1, constant_values=False)
    touches_outside = np.zeros(mask.shape, dtype=bool)
    touches_inside = np.zeros(mask.shape, dtype=bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbour = padded[1 + dy : padded.shape[0] - 1 + dy, 1 + dx : padded.shape[1] - 1 + dx]
        touches_outside |= ~neighbour
        touches_inside |= neighbour

    inner = mask & touches_outside & ~measured
    outer = ~mask & touches_inside & ~measured
    result = coverage.copy()
    result[inner] = np.clip(hull_coverage[inner], 0.52, 1.0)
    result[outer] = np.clip(hull_coverage[outer], 0.0, 0.48)
    return result


def _shape_loops(partition, component, padded_labels, padded_colours, palette, options, rank=None):
    """Traced loops for one component, in image coordinates."""
    y0, x0, y1, x1 = partition.boxes[component]
    rows = slice(y0, y1 + 2 * _MARGIN)
    columns = slice(x0, x1 + 2 * _MARGIN)

    component_crop = np.pad(partition.component_map, _MARGIN, constant_values=-1)[rows, columns]
    mask = component_crop == component
    hull_coverage = None
    if options.use_stacking:
        mask = _fill_holes(mask)
        if options.use_occlusion and rank is not None:
            mask, hull_coverage = _occluded_extension(mask, component_crop, rank, rank[component])

    labels = padded_labels[rows, columns]
    if options.use_coverage:
        coverage, measured = coverage_field(
            mask, labels, padded_colours[rows, columns], palette, report=True
        )
        if hull_coverage is not None:
            # Hidden edges have no antialiasing to read; their geometry is
            # the hull's, so take coverage from the hull itself.
            coverage = _blend_hidden_edges(coverage, measured, mask, hull_coverage)
    else:
        coverage = mask.astype(np.float64)

    # Sample (r, c) of the once-padded field is crop pixel (r-1, c-1), whose
    # centre sits at image (x0 - margin + c - 0.5, y0 - margin + r - 0.5).
    offset = np.array([x0 - _MARGIN - 0.5, y0 - _MARGIN - 0.5])
    loops = trace_loops(np.pad(coverage, 1))
    return [loop + offset for loop in loops], int(mask.sum())


def _partition_within_budget(labelling, view, options):
    """Coarsen until the region count fits the budget, rather than refuse.

    A detailed illustration can legitimately exceed the budget, and the
    useful answer is a slightly simpler trace that says so. Only input that
    stays over budget even when coarsened hard — pure noise — is refused.
    """
    if labelling.raw_noise > options.max_noise:
        raise VectorizeRejected(
            f"input is mostly noise (level {labelling.raw_noise:.0f}) — there is no structure to trace", 0
        )
    scale = 1.0
    for _ in range(7):
        partition = partition_labels(labelling, view, scale)
        count = len(partition.areas)
        if count <= options.max_regions:
            notes = ()
            if scale > 1.0:
                notes = (f"simplified to fit the {options.max_regions}-region budget "
                         f"(smallest regions merged at {scale:.0f}× the usual size)",)
            return partition, notes
        scale *= 2.0
    raise VectorizeRejected(
        f"input still produced {count} regions after coarsening — too noisy to trace", count
    )


def vectorize_v2(path_or_file, options=None):
    options = options or V2Options()
    rgb, scale, was_jpeg = _load(path_or_file, options.max_dimension)
    height, width = rgb.shape[:2]

    blend = options.blend_residual if options.use_blend_labels else -1.0
    view = _PartitionView(options, blend)
    view.blend_reach = options.jpeg_blend_reach if was_jpeg else 1
    labelling = label_image(rgb, view)
    partition, notes = _partition_within_budget(labelling, view, options)
    count = len(partition.areas)
    rgb = labelling.image

    palette = partition.palette.astype(np.float64) / 255.0
    padded_labels = np.pad(partition.label_map, _MARGIN, constant_values=-1)
    padded_colours = np.pad(rgb.astype(np.float64) / 255.0, ((_MARGIN, _MARGIN), (_MARGIN, _MARGIN), (0, 0)))

    settings = FitSettings(**{**options.fit.__dict__, "tolerance": options.tolerance})
    settings.greedy = not options.use_optimal_fit

    # Largest enclosed area first: anything inside a shape's filled holes
    # is smaller, so it is always painted on top.
    padded_components = np.pad(partition.component_map, _MARGIN, constant_values=-1)
    filled_areas = np.zeros(count, dtype=np.int64)
    for component in range(count):
        y0, x0, y1, x1 = partition.boxes[component]
        crop = padded_components[y0 : y1 + 2 * _MARGIN, x0 : x1 + 2 * _MARGIN] == component
        filled_areas[component] = int(_fill_holes(crop).sum()) if options.use_stacking else int(crop.sum())
    order = sorted(range(count), key=lambda component: (-filled_areas[component], component))
    rank = np.empty(count, dtype=np.int64)
    rank[np.array(order, dtype=np.int64)] = np.arange(count)

    shapes = []
    for component in order:
        loops, _ = _shape_loops(
            partition, component, padded_labels, padded_colours, palette, options, rank
        )
        if loops:
            shapes.append((component, loops))

    inverse = 1.0 / scale
    plan = []
    to_fit = []
    for component, loops in shapes:
        primitive = None
        if options.use_primitives and len(loops) == 1:
            primitive = detect_primitive(loops[0], options.tolerance)
        if primitive is not None:
            plan.append((component, primitive, None))
        else:
            plan.append((component, None, (len(to_fit), len(loops))))
            to_fit.extend(loops)

    # Every contour in the image is fitted in one shared batch.
    fitted_all = fit_loops(to_fit, settings)

    elements = []
    for component, primitive, span in plan:
        fill = to_hex(partition.palette[partition.component_label[component]])
        if primitive is not None:
            elements.append(primitive_element(primitive, fill, inverse))
            continue
        start, count = span
        fitted = fitted_all[start : start + count]
        data = path_data(fitted, inverse)
        if data:
            rule = ' fill-rule="evenodd"' if len(fitted) > 1 else ""
            elements.append(f'<path d="{data}" fill="{fill}"{rule}/>')

    document = build_document(elements, round(width * inverse), round(height * inverse))
    counts = count_editable_points(document)
    return VectorizeResult(
        svg=document,
        width=round(width * inverse),
        height=round(height * inverse),
        shape_count=counts["shapes"],
        path_segment_count=counts["anchors"],
        palette=[to_hex(colour) for colour in partition.palette],
        notes=notes,
    )


class _PartitionView:
    """Adapts V2Options to the field names build_partition reads."""

    def __init__(self, options, blend_residual):
        self.max_colors = options.max_colors
        self.min_palette_fraction = options.min_palette_fraction
        self.palette_merge_delta = options.palette_merge_delta
        self.pure_delta = options.pure_delta
        self.min_flat_coverage = options.min_flat_coverage
        self.blend_residual = blend_residual
        self.fallback_colors = options.fallback_colors
        self.max_fallback_colors = options.max_fallback_colors
        self.target_delta = options.target_delta
        self.min_region_area = options.min_region_area
        self.noisy_region_area = options.noisy_region_area
        self.keep_contrast = options.keep_contrast
        self.residual_delta = options.residual_delta
        self.min_missing_colour = options.min_missing_colour


__all__ = ["V2Options", "vectorize_v2"]
