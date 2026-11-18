"""Gradient fills: the bands of one smooth colour ramp become one shape.

A gradient has no flat colours, so the palette slices it into bands.
Neighbouring bands of a ramp meet without an edge: the colour step
between the pixels either side of their boundary is a small fraction of
the difference between the bands' colours. Real edges step most of that
difference at once. Bands joined by soft boundaries are grouped, and a
group becomes one shape if a gradient reproduces its pixels.

The gradient is a linear or radial ramp, optionally with radial glows
over it that fade to transparent, which is how designers draw a sun's
haze over a sky or a highlight on a button.
"""

from dataclasses import dataclass, field

import numpy as np

from ..quantize import srgb_to_lab
from .partition import Partition, _component_boxes

_SOFT_STEP = 8.0  # ΔE: largest mean pixel step across a soft boundary
_SOFT_RATIO = 0.4  # soft boundaries step less than this share of the band difference
_SOFT_CONTINUITY = 1.6  # ...and not much more than the steps inside the bands
_SOFT_FLOOR = 0.5  # ΔE allowed on top, for flat bands
_MAX_MEAN_ERROR = 2.5  # ΔE: a gradient must reproduce its pixels this closely on average
_MAX_P95_ERROR = 6.0
_INLIER_ERROR = 3.0
_MIN_AREA = 64
_MIN_LAYERED = 1500
_MIN_SPLIT_AREA = 1000
_MIN_RADIAL = 400
_SAMPLE = 20000
_SEARCH_SAMPLE = 3000
_BINS = 64
_STOP_TOLERANCE = 3.0  # sRGB levels
_MAX_STOPS = 8
_MAX_OVERLAYS = 3
_PRIOR_WEIGHT = 4.0  # pulls a ring's opacity towards the ring outside it when the base barely varies


@dataclass
class Gradient:
    kind: str  # "linear" or "radial"
    geometry: tuple  # linear: (x1, y1, x2, y2); radial: (cx, cy, r) — image pixels
    stops: list  # [(offset 0..1, (r, g, b) floats 0..255, opacity 0..1)]


@dataclass
class GradientFill:
    base: Gradient
    overlays: list = field(default_factory=list)  # radial, fading out; each drawn as a circle on the base


# ---------------------------------------------------------------- soft boundaries


def _neighbour_steps(component_map, image_lab):
    """ΔE between 4-neighbours, as (first ids, second ids, steps) for each axis."""
    axes = []
    for first, second, first_lab, second_lab in (
        (component_map[:, :-1], component_map[:, 1:], image_lab[:, :-1], image_lab[:, 1:]),
        (component_map[:-1], component_map[1:], image_lab[:-1], image_lab[1:]),
    ):
        axes.append((
            first.ravel().astype(np.int64),
            second.ravel().astype(np.int64),
            np.linalg.norm(first_lab - second_lab, axis=-1).ravel(),
        ))
    return axes


def _boundary_steps(component_map, image_lab):
    """Per adjacent component pair: (pairs (m, 2), mean ΔE step across it, boundary length),
    and per component the colour change per pixel inside it."""
    count = int(component_map.max()) + 1
    axes = _neighbour_steps(component_map, image_lab)
    # Steps along each axis, combined: a ramp's slope whatever its direction.
    per_axis = []
    for first, second, step in axes:
        same = first == second
        total = np.bincount(first[same], weights=step[same], minlength=count)
        per_axis.append(total / np.maximum(np.bincount(first[same], minlength=count), 1))
    internal = np.hypot(*per_axis)

    first, second, step = (np.concatenate(parts) for parts in zip(*axes))
    crossing = first != second
    low, high = np.minimum(first[crossing], second[crossing]), np.maximum(first[crossing], second[crossing])
    unique, inverse = np.unique(low * count + high, return_inverse=True)
    inverse = inverse.ravel()
    lengths = np.bincount(inverse)
    means = np.bincount(inverse, weights=step[crossing]) / np.maximum(lengths, 1)
    pairs = np.stack([unique // count, unique % count], axis=1)
    return pairs, means, lengths, internal


def soft_pairs(partition, image_lab, excluded_label=None):
    """Adjacent component pairs whose shared boundary has no edge.

    Across a band boundary in a gradient the colour steps about as much
    as it does inside the bands. Across an edge, even a blurred one, it
    steps far more than inside the flat regions either side.
    """
    pairs, steps, lengths, internal = _boundary_steps(partition.component_map, image_lab)
    palette_lab = srgb_to_lab(partition.palette.astype(np.float64) / 255.0)
    labels = partition.component_label
    difference = np.linalg.norm(palette_lab[labels[pairs[:, 0]]] - palette_lab[labels[pairs[:, 1]]], axis=1)
    within = np.maximum(internal[pairs[:, 0]], internal[pairs[:, 1]])
    soft = (
        (steps <= _SOFT_CONTINUITY * within + _SOFT_FLOOR)
        & (steps <= _SOFT_RATIO * difference)
        & (steps <= _SOFT_STEP)
        & (lengths >= 3)
    )
    if excluded_label is not None:
        soft &= (labels[pairs[:, 0]] != excluded_label) & (labels[pairs[:, 1]] != excluded_label)
    return pairs[soft]


def _neighbours(pairs, count):
    linked = [set() for _ in range(count)]
    for first, second in pairs.tolist():
        linked[first].add(second)
        linked[second].add(first)
    return linked


def _connected(members, linked):
    """Split a set of components into its connected parts."""
    remaining, parts = set(members), []
    while remaining:
        seed = remaining.pop()
        part, frontier = {seed}, [seed]
        while frontier:
            node = frontier.pop()
            for other in linked[node] & remaining:
                remaining.discard(other)
                part.add(other)
                frontier.append(other)
        parts.append(part)
    return parts


# ---------------------------------------------------------------- ramps


def _binned(position, values, robust):
    """Per-bin centre and median (or mean) value along position in [0, 1]."""
    bins = np.minimum((np.clip(position, 0.0, 1.0) * _BINS).astype(np.int64), _BINS - 1)
    filled = np.unique(bins)
    if robust:
        summary = np.empty((len(filled), values.shape[1]))
        for channel in range(values.shape[1]):
            order = np.lexsort((values[:, channel], bins))
            starts = np.searchsorted(bins[order], filled)
            ends = np.append(starts[1:], len(bins))
            summary[:, channel] = values[order[(starts + ends - 1) // 2], channel]
    else:
        counts = np.bincount(bins, minlength=_BINS)[filled]
        summary = np.stack(
            [np.bincount(bins, weights=values[:, c], minlength=_BINS)[filled] for c in range(values.shape[1])], 1
        ) / counts[:, None]
    centres = (filled + 0.5) / _BINS
    centres[0], centres[-1] = 0.0, 1.0
    return centres, summary


def _simplify(centres, values, tolerance):
    """Indices of the fewest points whose piecewise-linear ramp stays within tolerance."""
    keep = {0, len(centres) - 1}
    pending = [(0, len(centres) - 1)]
    while pending and len(keep) < _MAX_STOPS:
        first, last = pending.pop()
        if last - first < 2:
            continue
        inner = np.arange(first + 1, last)
        weight = (centres[inner] - centres[first]) / max(centres[last] - centres[first], 1e-9)
        line = values[first] + weight[:, None] * (values[last] - values[first])
        deviation = np.abs(values[inner] - line).max(axis=1)
        worst = int(np.argmax(deviation))
        if deviation[worst] > tolerance:
            split = int(inner[worst])
            keep.add(split)
            pending.extend([(first, split), (split, last)])
    return sorted(keep)


def _opaque_stops(position, colours, robust):
    centres, means = _binned(position, colours, robust)
    if len(centres) == 1:
        return [(0.0, tuple(means[0]), 1.0)]
    return [(float(centres[i]), tuple(means[i]), 1.0) for i in _simplify(centres, means, _STOP_TOLERANCE)]


def _interp(position, stops):
    offsets = np.array([stop[0] for stop in stops])
    colours = np.array([stop[1] for stop in stops])
    opacity = np.array([stop[2] for stop in stops])
    colour = np.stack([np.interp(position, offsets, colours[:, c]) for c in range(3)], axis=1)
    return colour, np.interp(position, offsets, opacity)


def _position(gradient, points):
    if gradient.kind == "linear":
        x1, y1, x2, y2 = gradient.geometry
        axis = np.array([x2 - x1, y2 - y1])
        return ((points - (x1, y1)) @ axis) / max(float(axis @ axis), 1e-12)
    cx, cy, radius = gradient.geometry
    return np.hypot(points[:, 0] - cx, points[:, 1] - cy) / max(radius, 1e-12)


def paint(fill, points):
    """Colour a GradientFill gives each point, as an SVG renderer would."""
    colour, _ = _interp(_position(fill.base, points), fill.base.stops)
    for overlay in fill.overlays:
        position = _position(overlay, points)
        top, opacity = _interp(position, overlay.stops)
        opacity = np.where(position <= 1.0, opacity, 0.0)
        colour = opacity[:, None] * top + (1.0 - opacity[:, None]) * colour
    return colour


def _error(predicted, actual_lab):
    return np.linalg.norm(srgb_to_lab(np.clip(predicted, 0, 255) / 255.0) - actual_lab, axis=1)


# ---------------------------------------------------------------- single gradients


def _linear_at(points, colours, angle, robust):
    direction = np.array([np.cos(angle), np.sin(angle)])
    origin = points.mean(axis=0)
    t = (points - origin) @ direction
    low, high = float(t.min()), float(t.max())
    stops = _opaque_stops((t - low) / max(high - low, 1e-9), colours, robust)
    start, end = origin + low * direction, origin + high * direction
    return Gradient("linear", (start[0], start[1], end[0], end[1]), stops)


def _radial_at(points, colours, centre, robust):
    t = np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1])
    low, high = float(t.min()), float(max(t.max(), 1e-6))
    stops = _opaque_stops((t - low) / max(high - low, 1e-9), colours, robust)
    # Offsets are fractions of the radius, measured from the centre.
    stops = [(low / high + offset * (high - low) / high, colour, 1.0) for offset, colour, _ in stops]
    return Gradient("radial", (centre[0], centre[1], high), stops)


def _scored(gradient, points, lab):
    return gradient, _error(paint(GradientFill(gradient), points), lab)


def _few(count):
    """A fixed subsample for searches; the chosen fit is redone on every point."""
    if count <= _SEARCH_SAMPLE:
        return np.arange(count)
    return np.sort(np.random.default_rng(5).choice(count, _SEARCH_SAMPLE, replace=False))


def _fit_linear(points, colours, lab, robust=False):
    design = np.column_stack([points - points.mean(axis=0), np.ones(len(points))])
    coefficients, *_ = np.linalg.lstsq(design, colours, rcond=None)
    _, _, right = np.linalg.svd(coefficients[:2].T)
    base = float(np.arctan2(right[0, 1], right[0, 0]))
    few = _few(len(points))

    def score(angle):
        return float(_scored(_linear_at(points[few], colours[few], angle, robust), points[few], lab[few])[1].mean())

    angle = min((base + step for step in np.radians(np.arange(-8, 9, 2))), key=score)
    return _scored(_linear_at(points, colours, angle, robust), points, lab)


def _search_centre(score, start, span):
    """Pattern search for the centre that minimises score(centre)."""
    best_centre, best = np.asarray(start, dtype=np.float64), score(start)
    step = 0.08 * span
    for _ in range(10):
        moved = False
        for offset in ((step, 0), (-step, 0), (0, step), (0, -step)):
            trial = best_centre + offset
            value = score(trial)
            if value < best:
                best_centre, best, moved = trial, value, True
        if not moved:
            step /= 2.0
            if step < 0.5:
                break
    return best_centre


def _fit_radial(points, colours, lab, robust=False):
    design = np.column_stack([(points ** 2).sum(axis=1), points, np.ones(len(points))])
    coefficients, *_ = np.linalg.lstsq(design, colours, rcond=None)
    curvature = coefficients[0]
    strongest = int(np.argmax(np.abs(curvature)))
    if abs(curvature[strongest]) < 1e-9:
        return None
    start = -coefficients[1:3, strongest] / (2.0 * curvature[strongest])
    span = float(np.ptp(points, axis=0).max())
    if not np.all(np.isfinite(start)) or np.linalg.norm(start - points.mean(axis=0)) > 3.0 * span:
        return None

    few = _few(len(points))

    def score(centre):
        return float(_scored(_radial_at(points[few], colours[few], centre, robust), points[few], lab[few])[1].mean())

    # The search only refines the centre; one far off at the start won't come good.
    if score(start) > 4.0 * _MAX_MEAN_ERROR:
        return None
    return _scored(_radial_at(points, colours, _search_centre(score, start, span), robust), points, lab)


def _fit_single(points, colours, lab, robust=False, radial=True):
    """Best of linear and radial; radial must earn its extra freedom.

    Small regions skip the radial search: over a few hundred pixels the
    two look alike, and small soft regions are numerous in detailed art.
    """
    linear = _fit_linear(points, colours, lab, robust)
    if _acceptable(linear[1]) or not radial or len(points) < _MIN_RADIAL:
        return linear
    radial = _fit_radial(points, colours, lab, robust)
    if radial is not None and radial[1].mean() < 0.8 * linear[1].mean():
        return radial
    return linear


def _acceptable(error):
    return float(error.mean()) <= _MAX_MEAN_ERROR and float(np.percentile(error, 95)) <= _MAX_P95_ERROR


# ---------------------------------------------------------------- layered: base plus glows


def _robust_base(points, colours, lab):
    """A ramp fitted to the pixels it explains, ignoring a glow or highlight on top."""
    gradient, error = _fit_single(points, colours, lab, robust=True)
    for round_index in range(3):
        threshold = max(_INLIER_ERROR, float(np.percentile(error, 50))) if round_index == 0 else _INLIER_ERROR
        keep = error <= threshold
        if keep.sum() < _MIN_AREA:
            break
        gradient, _ = _fit_single(points[keep], colours[keep], lab[keep], robust=True)
        error = _error(paint(GradientFill(gradient), points), lab)
    return gradient


def _glow_profile(radius, actual, under, outer):
    """Opacity and colour per ring of a glow, from the outside in.

    In each ring actual = opacity * colour + (1 - opacity) * under. The
    base varies around a ring, which separates opacity from colour; where
    it barely varies, the ring outside it decides.
    """
    bins = np.minimum((radius / outer * _BINS).astype(np.int64), _BINS - 1)
    centres = (np.arange(_BINS) + 0.5) / _BINS
    counts = np.bincount(bins, minlength=_BINS).astype(np.float64)

    def per_bin(values):
        return np.bincount(bins, weights=values, minlength=_BINS)

    mean_actual = np.stack([per_bin(actual[:, c]) for c in range(3)], 1) / np.maximum(counts, 1)[:, None]
    mean_under = np.stack([per_bin(under[:, c]) for c in range(3)], 1) / np.maximum(counts, 1)[:, None]
    # Centred sums per ring: Σ(a - ā)(s - s̄) = Σas - n·ā·s̄, pooled over channels.
    covariance = per_bin((actual * under).sum(axis=1)) - counts * (mean_actual * mean_under).sum(axis=1)
    variance = per_bin((under * under).sum(axis=1)) - counts * (mean_under * mean_under).sum(axis=1)

    opacity = np.zeros(_BINS)
    colour = np.zeros((_BINS, 3))
    keep_through = 1.0  # 1 - opacity outside the glow
    previous_colour = None
    for index in range(_BINS - 1, -1, -1):
        if counts[index] >= 8:
            prior = _PRIOR_WEIGHT * counts[index]
            keep_through = float(np.clip(
                (covariance[index] + prior * keep_through) / (max(variance[index], 0.0) + prior), 0.0, 1.0
            ))
            alpha = 1.0 - keep_through
            if alpha > 0.02:
                previous_colour = np.clip((mean_actual[index] - keep_through * mean_under[index]) / alpha, 0, 255)
        opacity[index] = 1.0 - keep_through
        colour[index] = previous_colour if previous_colour is not None else actual.mean(axis=0)
    # Where a glow is nearly transparent its colour is unknown; take the nearest known one.
    known = np.flatnonzero(opacity > 0.02)
    if len(known):
        for index in range(_BINS):
            if opacity[index] <= 0.02:
                colour[index] = colour[known[np.argmin(np.abs(known - index))]]
    centres[0], centres[-1] = 0.0, 1.0
    # A glow that has almost faded at its rim fades fully, or its edge would show.
    if opacity[-1] < 0.1:
        opacity[-1] = 0.0
    values = np.column_stack([colour * opacity[:, None], 255.0 * opacity])
    return [(float(centres[i]), tuple(colour[i]), float(opacity[i])) for i in _simplify(centres, values, _STOP_TOLERANCE)]


def _glow_at(points, actual, under, centre, outer):
    radius = np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1])
    inside = radius < outer
    stops = _glow_profile(radius[inside], actual[inside], under[inside], outer)
    return Gradient("radial", (centre[0], centre[1], outer), stops)


def _fit_glow(points, colours, lab, fill, error):
    """A radial glow fading to transparent that explains the pixels the fill misses."""
    missed = error > _INLIER_ERROR
    if missed.sum() < _MIN_AREA:
        return None
    weights = error[missed]
    start = (points[missed] * weights[:, None]).sum(axis=0) / weights.sum()
    under = paint(fill, points)
    few = _few(len(points))

    def glow_for(centre, chosen):
        outer = float(np.percentile(np.hypot(*(points[missed] - centre).T), 99)) + 1.0
        return _glow_at(points[chosen], colours[chosen], under[chosen], centre, outer)

    def score(centre):
        trial = GradientFill(fill.base, fill.overlays + [glow_for(centre, few)])
        return float(_error(paint(trial, points[few]), lab[few]).mean())

    span = float(np.ptp(points[missed], axis=0).max()) + 1.0
    return glow_for(_search_centre(score, start, span), np.arange(len(points)))


def _fit_layered(points, colours, lab):
    fill = GradientFill(_robust_base(points, colours, lab))
    error = _error(paint(fill, points), lab)
    for _ in range(_MAX_OVERLAYS):
        if _acceptable(error):
            return fill, error
        glow = _fit_glow(points, colours, lab, fill, error)
        if glow is None:
            break
        trial = GradientFill(fill.base, fill.overlays + [glow])
        trial_error = _error(paint(trial, points), lab)
        if trial_error.mean() >= 0.9 * error.mean():
            break
        fill, error = trial, trial_error
    return fill, error


# ---------------------------------------------------------------- grouping


class _Pixels:
    """Interior pixels of every component, for fitting."""

    def __init__(self, component_map, rgb, image_lab):
        padded = np.pad(component_map, 1, mode="edge")
        interior = np.ones(component_map.shape, dtype=bool)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            interior &= padded[1 + dy : padded.shape[0] - 1 + dy, 1 + dx : padded.shape[1] - 1 + dx] == component_map
        rows, columns = np.nonzero(interior)
        self.owner = component_map[rows, columns]
        self.points = np.column_stack([columns + 0.5, rows + 0.5])
        self.rgb = rgb[rows, columns].astype(np.float64)
        self.lab = image_lab[rows, columns]

    def select(self, members):
        chosen = np.flatnonzero(np.isin(self.owner, list(members)))
        if len(chosen) > _SAMPLE:
            chosen = np.sort(np.random.default_rng(11).choice(chosen, _SAMPLE, replace=False))
        return self.points[chosen], self.rgb[chosen], self.lab[chosen], self.owner[chosen]


def _component_errors(error, owners):
    ids, inverse = np.unique(owners, return_inverse=True)
    inverse = inverse.ravel()
    return dict(zip(ids.tolist(), (np.bincount(inverse, weights=error) / np.bincount(inverse)).tolist()))


def _inlier_components(members, pixels, linked):
    """Largest connected subset one plain gradient explains, refitting without the misfits."""
    current = set(members)
    for round_index in range(4):
        if len(current) < 2:
            return None, None
        points, colours, lab, owners = pixels.select(current)
        if len(points) < _MIN_AREA:
            return None, None
        # The caller already tried a radial fit on the whole group.
        gradient, error = _fit_single(points, colours, lab, radial=round_index > 0)
        if _acceptable(error):
            return current, GradientFill(gradient)
        errors = _component_errors(error, owners)
        inliers = {member for member in current if errors.get(member, 0.0) <= _MAX_MEAN_ERROR}
        if not inliers or inliers == current:
            return None, None
        current = max(_connected(inliers, linked), key=len)
    return None, None


def _fit_group(members, pixels, linked):
    """(members, GradientFill) for the part of a soft group a gradient explains, or (None, None)."""
    points, colours, lab, _ = pixels.select(members)
    if len(points) < _MIN_AREA:
        return None, None
    gradient, error = _fit_single(points, colours, lab)
    if _acceptable(error):
        return set(members), GradientFill(gradient)
    # Glows are worth searching for only where a ramp already explains most of the group.
    if len(points) >= _MIN_LAYERED and float(np.median(error)) <= _INLIER_ERROR:
        fill, error = _fit_layered(points, colours, lab)
        if _acceptable(error):
            return set(members), fill
    return _inlier_components(members, pixels, linked)


def _centres(component_map, count):
    rows, columns = np.indices(component_map.shape)
    flat = component_map.ravel()
    areas = np.maximum(np.bincount(flat, minlength=count), 1)
    return np.column_stack([
        np.bincount(flat, weights=columns.ravel(), minlength=count) / areas,
        np.bincount(flat, weights=rows.ravel(), minlength=count) / areas,
    ])


def _halves(group, centres, areas):
    """Split components across the group's long axis, at the area-weighted median."""
    members = np.array(sorted(group))
    points = centres[members]
    weights = areas[members].astype(np.float64)
    offset = points - np.average(points, axis=0, weights=weights)
    _, _, axes = np.linalg.svd(offset * np.sqrt(weights)[:, None], full_matrices=False)
    along = offset @ axes[0]
    order = np.argsort(along)
    cut = int(np.searchsorted(np.cumsum(weights[order]), 0.5 * weights.sum())) + 1
    cut = min(max(cut, 1), len(members) - 1)
    return set(members[order[:cut]].tolist()), set(members[order[cut:]].tolist())


def find_gradients(partition, rgb, image_lab, excluded_label=None):
    """Groups of components that are one gradient. Returns [(members, GradientFill)]."""
    pairs = soft_pairs(partition, image_lab, excluded_label)
    if not len(pairs):
        return []
    linked = _neighbours(pairs, len(partition.areas))
    pixels = _Pixels(partition.component_map, rgb, image_lab)

    centres = _centres(partition.component_map, len(partition.areas))
    found = []
    queue = [part for part in _connected({int(node) for node in np.unique(pairs)}, linked) if len(part) >= 2]
    while queue:
        group = queue.pop()
        if int(partition.areas[list(group)].sum()) < _MIN_AREA:
            continue
        members, fill = _fit_group(group, pixels, linked)
        if members is None:
            # A field of several gradients: try each half on its own.
            if len(group) >= 4 and int(partition.areas[list(group)].sum()) >= _MIN_SPLIT_AREA:
                for half in _halves(group, centres, partition.areas):
                    queue.extend(part for part in _connected(half, linked) if len(part) >= 2)
            continue
        found.append((members, fill))
        queue.extend(part for part in _connected(group - members, linked) if len(part) >= 2)
    return found


def _fringe_owner(colour, sides, tolerance=0.06):
    """The side a thin region's colour is mostly made of, if it's a blend of two sides."""
    best, owner = tolerance, None
    for first in range(len(sides)):
        for second in range(first + 1, len(sides)):
            (owner_a, colour_a), (owner_b, colour_b) = sides[first], sides[second]
            span = colour_a - colour_b
            reach = float(span @ span)
            if reach < 1e-6:
                continue
            mix = float((colour - colour_b) @ span) / reach
            if not 0.05 < mix < 0.95:
                continue
            residual = float(np.linalg.norm(colour - (colour_b + mix * span)))
            if residual < best:
                best, owner = residual, owner_a if mix >= 0.5 else owner_b
    return owner


def _fringes(partition, groups, rgb, image_lab):
    """Thin regions along a gradient's edge that are antialiasing against it.

    Edge pixels next to a gradient blend with a colour that isn't in the
    palette, so they can't be explained as a blend when labelled and
    survive as slivers. Against the gradient's own colour there, they are.
    Returns {component: owner}, where an owner is a component or ("group", index).
    """
    pairs, _, lengths, _ = _boundary_steps(partition.component_map, image_lab)
    in_group = {member: index for index, (members, _) in enumerate(groups) for member in members}
    touching = [dict() for _ in range(len(partition.areas))]
    for (first, second), length in zip(pairs.tolist(), lengths.tolist()):
        touching[first][second] = length
        touching[second][first] = length

    flat = partition.component_map.ravel()
    sums = np.stack([np.bincount(flat, weights=rgb[..., c].ravel(), minlength=len(partition.areas)) for c in range(3)], 1)
    rows, columns = np.indices(partition.component_map.shape)
    centre_x = np.bincount(flat, weights=columns.ravel() + 0.5, minlength=len(partition.areas))
    centre_y = np.bincount(flat, weights=rows.ravel() + 0.5, minlength=len(partition.areas))
    palette = partition.palette.astype(np.float64) / 255.0

    owners = {}
    for component, others in enumerate(touching):
        area = int(partition.areas[component])
        if component in in_group or not others or area > 0.6 * sum(others.values()):
            continue
        group_sides = {in_group[other] for other in others if other in in_group}
        if not group_sides:
            continue
        centre = np.array([[centre_x[component] / area, centre_y[component] / area]])
        sides = [(("group", index), paint(groups[index][1], centre)[0] / 255.0) for index in sorted(group_sides)]
        sides += [(other, palette[partition.component_label[other]]) for other in others if other not in in_group]
        owner = _fringe_owner(sums[component] / area / 255.0, sides)
        if owner is not None:
            owners[component] = owner
    return owners


def merge_gradients(partition, rgb, image_lab, excluded_label=None):
    """Partition with each gradient group as one component, and the fill per component.

    fills[i] is None for a flat colour, or the GradientFill of component i.
    """
    groups = find_gradients(partition, rgb, image_lab, excluded_label)
    if not groups:
        return partition, [None] * len(partition.areas)

    count = len(partition.areas)
    target = np.arange(count)
    fill_of = {}
    roots = []
    for members, fill in groups:
        ordered = sorted(members)
        target[ordered] = ordered[0]
        fill_of[ordered[0]] = fill
        roots.append(ordered[0])
    fringes = _fringes(partition, groups, rgb, image_lab)
    for component, owner in fringes.items():
        if isinstance(owner, tuple):
            target[component] = roots[owner[1]]
        elif owner not in fringes:
            target[component] = owner
    kept = np.unique(target)
    renumber = np.full(count, -1, dtype=np.int64)
    renumber[kept] = np.arange(len(kept))
    new_id = renumber[target]

    component_map = new_id[partition.component_map].astype(np.int32)
    # A group keeps the label of its largest member, so coverage reads a typical colour.
    largest = {}
    for component in range(count):
        owner = int(new_id[component])
        if owner not in largest or partition.areas[component] > partition.areas[largest[owner]]:
            largest[owner] = component
    merged = Partition(
        palette=partition.palette,
        label_map=partition.label_map,
        component_map=component_map,
        component_label=np.array([partition.component_label[largest[i]] for i in range(len(kept))], dtype=np.int32),
        areas=np.bincount(component_map.ravel(), minlength=len(kept)),
        boxes=_component_boxes(component_map, len(kept)),
        palette_mode=partition.palette_mode,
    )
    return merged, [fill_of.get(int(old)) for old in kept]
