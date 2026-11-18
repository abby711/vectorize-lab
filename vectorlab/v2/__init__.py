"""vectorlab v2 — sub-pixel, globally fitted, primitive-aware tracing.

    decode (transparency composited over a key colour)
           → detect palette → blend-aware labels → partition (merge, never drop)
           → bands of one gradient grouped into a gradient-filled shape
           → paint order by enclosed area, holes filled
           → coverage field → marching squares → primitive or optimal spline
           → SVG, leaving out the key colour's regions

Every idea is a switch on V2Options so each one's contribution can be
measured on its own.
"""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from ..metrics import count_editable_points
from ..pipeline import VectorizeRejected, VectorizeResult
from ..quantize import srgb_to_lab
from .contour import coverage_field, signed_area, trace_loops
from .emit import build_document, glow_element, gradient_element, path_data, primitive_element, to_hex
from .fitting import FitSettings, fit_loops
from .gradients import merge_gradients
from .partition import force_clear, label_image, label_partition, partition_labels
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
    use_gradients: bool = True


@dataclass
class _Source:
    rgb: np.ndarray
    alpha: np.ndarray = None  # 0..1, only when the image has transparency
    key: np.ndarray = None  # the colour transparency was composited over
    scale: float = 1.0
    was_jpeg: bool = False


_OPAQUE = 250


def _key_colour(rgba):
    """The colour furthest from everything opaque in the image.

    Composited over it, a partly transparent edge pixel becomes a blend of
    two palette colours, which the labeller and the coverage field
    already know how to unmix.
    """
    pixels = rgba[rgba[..., 3] >= 128][:, :3]
    if not len(pixels):
        return np.array([255, 0, 255], dtype=np.uint8)
    chosen = np.random.default_rng(0).choice(len(pixels), min(20000, len(pixels)), replace=False)
    sample = srgb_to_lab(pixels[chosen].astype(np.float64) / 255.0)
    levels = np.arange(0, 256, 51)
    candidates = np.stack(np.meshgrid(levels, levels, levels, indexing="ij"), axis=-1).reshape(-1, 3)
    candidate_lab = srgb_to_lab(candidates / 255.0)
    # A low quantile, not the minimum, so a few stray pixels can't veto a key.
    clearance = [
        float(np.quantile(np.linalg.norm(sample - lab, axis=1), 0.002)) for lab in candidate_lab
    ]
    return candidates[int(np.argmax(clearance))].astype(np.uint8)


def _resize(array, size):
    return np.array(Image.fromarray(array).resize(size, Image.BOX))


def _load(path_or_file, max_dimension):
    opened = Image.open(path_or_file)
    was_jpeg = opened.format == "JPEG"
    has_alpha = "A" in opened.getbands() or "transparency" in opened.info
    rgba = np.array(opened.convert("RGBA")) if has_alpha else None
    if rgba is not None and (rgba[..., 3] < _OPAQUE).any():
        key = _key_colour(rgba)
        opacity = rgba[..., 3:4].astype(np.float64) / 255.0
        rgb = np.round(rgba[..., :3] * opacity + key * (1.0 - opacity)).astype(np.uint8)
        alpha = rgba[..., 3]
    else:
        key, alpha = None, None
        rgb = rgba[..., :3].copy() if rgba is not None else np.array(opened.convert("RGB"))

    scale = 1.0
    height, width = rgb.shape[:2]
    if max_dimension and max(height, width) > max_dimension:
        scale = max_dimension / float(max(height, width))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        # Box filtering, not Lanczos: coverage recovery assumes an edge
        # pixel is an area average, and Lanczos ringing paints false halos.
        rgb = _resize(rgb, size)
        alpha = _resize(alpha, size) if alpha is not None else None
    alpha = alpha.astype(np.float64) / 255.0 if alpha is not None else None
    return _Source(rgb, alpha, key, scale, was_jpeg)


def _fill_holes(mask, blocked=None):
    """Mask plus every enclosed pocket. Background is 8-connected, matching the tracer.

    A pocket holding any `blocked` pixel stays open: filling it would
    paint over a transparent hole.
    """
    outside, values = label_partition((~mask).astype(np.int8), connectivity=8)
    open_ids = np.unique(
        np.concatenate((outside[0], outside[-1], outside[:, 0], outside[:, -1]))
    )
    if blocked is not None and blocked.any():
        open_ids = np.union1d(open_ids, np.unique(outside[blocked & ~mask]))
    pocket = ~mask & ~np.isin(outside, open_ids)
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


@dataclass
class _Canvas:
    """Image-wide arrays, padded by _MARGIN so every crop has a border."""

    labels: np.ndarray
    colours: np.ndarray
    components: np.ndarray
    clear: np.ndarray  # transparent pixels; all False for an opaque image
    palette: np.ndarray  # 0..1 floats
    rank: np.ndarray = None  # paint order; -1 for components never painted

    def crop(self, box):
        y0, x0, y1, x1 = box
        return slice(y0, y1 + 2 * _MARGIN), slice(x0, x1 + 2 * _MARGIN)


def _make_canvas(partition, rgb, clear_index):
    clear = (
        partition.label_map == clear_index
        if clear_index is not None
        else np.zeros(partition.label_map.shape, dtype=bool)
    )
    return _Canvas(
        labels=np.pad(partition.label_map, _MARGIN, constant_values=-1),
        colours=np.pad(rgb.astype(np.float64) / 255.0, ((_MARGIN, _MARGIN), (_MARGIN, _MARGIN), (0, 0))),
        components=np.pad(partition.component_map, _MARGIN, constant_values=-1),
        clear=np.pad(clear, _MARGIN, constant_values=False),
        palette=partition.palette.astype(np.float64) / 255.0,
    )


def _paint_rank(partition, canvas, options, clear_components):
    """Largest enclosed area first: anything inside a shape's filled holes is smaller, so lands on top."""
    count = len(partition.areas)
    filled_areas = np.zeros(count, dtype=np.int64)
    for component in range(count):
        rows, columns = canvas.crop(partition.boxes[component])
        crop = canvas.components[rows, columns] == component
        if options.use_stacking:
            crop = _fill_holes(crop, canvas.clear[rows, columns])
        filled_areas[component] = int(crop.sum())
    order = [
        component
        for component in sorted(range(count), key=lambda component: (-filled_areas[component], component))
        if not clear_components[component]
    ]
    rank = np.full(count, -1, dtype=np.int64)
    rank[np.array(order, dtype=np.int64)] = np.arange(len(order))
    return order, rank


def _shape_loops(partition, component, canvas, options):
    """Traced loops for one component, in image coordinates."""
    y0, x0 = partition.boxes[component][:2]
    rows, columns = canvas.crop(partition.boxes[component])

    component_crop = canvas.components[rows, columns]
    mask = component_crop == component
    hull_coverage = None
    if options.use_stacking:
        mask = _fill_holes(mask, canvas.clear[rows, columns])
        if options.use_occlusion and canvas.rank is not None:
            mask, hull_coverage = _occluded_extension(
                mask, component_crop, canvas.rank, canvas.rank[component]
            )

    labels = canvas.labels[rows, columns]
    if options.use_coverage:
        coverage, measured = coverage_field(
            mask, labels, canvas.colours[rows, columns], canvas.palette, report=True
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
    # A sliver whose every pixel is under half covered traces to nothing visible.
    loops = [loop for loop in trace_loops(np.pad(coverage, 1)) if abs(signed_area(loop)) >= 0.5]
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
    source = _load(path_or_file, options.max_dimension)
    height, width = source.rgb.shape[:2]
    scale = source.scale

    blend = options.blend_residual if options.use_blend_labels else -1.0
    view = _PartitionView(options, blend)
    view.blend_reach = options.jpeg_blend_reach if source.was_jpeg else 1
    view.key_colour = source.key
    labelling = label_image(source.rgb, view)
    clear_index = force_clear(labelling, source.alpha, source.key) if source.key is not None else None
    partition, notes = _partition_within_budget(labelling, view, options)
    fills = [None] * len(partition.areas)
    if options.use_gradients:
        partition, fills = merge_gradients(partition, labelling.image, labelling.image_lab, clear_index)

    settings = FitSettings(**{**options.fit.__dict__, "tolerance": options.tolerance})
    settings.greedy = not options.use_optimal_fit

    canvas = _make_canvas(partition, labelling.image, clear_index)
    clear_components = partition.component_label == (clear_index if clear_index is not None else -1)
    order, canvas.rank = _paint_rank(partition, canvas, options, clear_components)

    shapes = []
    for component in order:
        loops, _ = _shape_loops(partition, component, canvas, options)
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
    defs = []
    flat_colours = []

    def gradient_url(gradient):
        ident = f"g{len(defs) + 1}"
        defs.append(gradient_element(gradient, ident, inverse))
        return f"url(#{ident})"

    for component, primitive, span in plan:
        gradient_fill = fills[component]
        fill = to_hex(partition.palette[partition.component_label[component]])
        if gradient_fill is not None:
            fill = gradient_url(gradient_fill.base)
        elif fill not in flat_colours:
            flat_colours.append(fill)
        if primitive is not None:
            elements.append(primitive_element(primitive, fill, inverse))
        else:
            start, count = span
            fitted = fitted_all[start : start + count]
            data = path_data(fitted, inverse)
            if data:
                rule = ' fill-rule="evenodd"' if len(fitted) > 1 else ""
                elements.append(f'<path d="{data}" fill="{fill}"{rule}/>')
        for glow in gradient_fill.overlays if gradient_fill is not None else ():
            elements.append(glow_element(glow, gradient_url(glow), inverse))

    document = build_document(elements, round(width * inverse), round(height * inverse), defs)
    counts = count_editable_points(document)
    return VectorizeResult(
        svg=document,
        width=round(width * inverse),
        height=round(height * inverse),
        shape_count=counts["shapes"],
        path_segment_count=counts["anchors"],
        palette=flat_colours,
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
