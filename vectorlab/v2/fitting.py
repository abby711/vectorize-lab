"""Stage 3: the fewest curves that stay within tolerance, chosen globally.

VTracer decides where curves break with local rules — inflection points
and accumulated angle — and v1 with simplify-then-connect. Both are
greedy: each decision is made without knowing what it costs later.

Here every contour is split at its corners into runs. Along each run a
dynamic program considers every candidate breakpoint and finds the set
with the fewest segments whose curves all stay within tolerance of the
traced points: the global optimum over the candidates.

Two details matter as much as the search:

- At a smooth breakpoint one tangent is estimated and shared by both
  segments, so joins are smooth by construction.
- At a corner the tangent is left *free* and solved for with the
  control points. A sharp tip has no well-defined tangent to estimate,
  and guessing one costs extra anchors.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Segment:
    kind: str  # "line" or "cubic"
    start: np.ndarray
    control_1: np.ndarray
    control_2: np.ndarray
    end: np.ndarray


@dataclass
class FitSettings:
    tolerance: float = 0.6
    corner_angle_degrees: float = 55.0
    corner_window: float = 3.0
    tangent_window: float = 3.0
    candidate_tolerance: float = 0.3
    max_handle_ratio: float = 1.0
    samples: int = 28
    iterations: int = 2
    seams: int = 4
    max_span: int = 48
    max_turn_degrees: float = 170.0
    greedy: bool = False


# ---------------------------------------------------------------- geometry


def _dedupe(points):
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6
    points = points[keep]
    if len(points) > 1 and np.linalg.norm(points[0] - points[-1]) <= 1e-6:
        points = points[:-1]
    return points


def _cumulative_length(points, closed):
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    lengths = np.concatenate(([0.0], np.cumsum(steps)))
    total = lengths[-1] + (np.linalg.norm(points[0] - points[-1]) if closed else 0.0)
    return lengths, total


def _douglas_peucker(points, epsilon):
    """Indices kept by Douglas–Peucker on an open polyline."""
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        chord = points[last] - points[first]
        length = np.hypot(chord[0], chord[1])
        offsets = points[first + 1 : last] - points[first]
        if length < 1e-12:
            distances = np.hypot(offsets[:, 0], offsets[:, 1])
        else:
            distances = np.abs(offsets[:, 0] * chord[1] - offsets[:, 1] * chord[0]) / length
        worst = int(np.argmax(distances))
        if distances[worst] > epsilon:
            split = first + 1 + worst
            keep[split] = True
            stack.append((first, split))
            stack.append((split, last))
    return np.flatnonzero(keep)


def _tangent(points, lengths, index, window):
    """Unit tangent by local quadratic regression in arclength."""
    centre = lengths[index]
    near = np.flatnonzero(np.abs(lengths - centre) <= window)
    if len(near) < 3:
        near = np.arange(max(0, index - 2), min(len(points), index + 3))
    offset = lengths[near] - centre
    if len(near) >= 3 and np.ptp(offset) > 1e-9:
        design = np.stack([np.ones_like(offset), offset, offset * offset], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, points[near], rcond=None)
        direction = coefficients[1]
    else:
        direction = points[near[-1]] - points[near[0]]
    norm = np.hypot(direction[0], direction[1])
    if norm < 1e-12:
        direction = points[min(index + 1, len(points) - 1)] - points[max(index - 1, 0)]
        norm = max(np.hypot(direction[0], direction[1]), 1e-12)
    return direction / norm


# ---------------------------------------------------------------- corners


def find_corners(points, settings):
    """Indices of sharp corners on a closed loop, after non-maximum suppression."""
    count = len(points)
    lengths, perimeter = _cumulative_length(points, closed=True)
    window = min(settings.corner_window, perimeter / 8.0)
    if count < 6 or window <= 0:
        return np.zeros(0, dtype=np.int64)

    extended = np.concatenate((lengths - perimeter, lengths, lengths + perimeter))
    back = np.searchsorted(extended, lengths + perimeter - window) % count
    forward = np.searchsorted(extended, lengths + perimeter + window) % count

    incoming = points - points[back]
    outgoing = points[forward] - points
    norms = np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
    cosine = np.sum(incoming * outgoing, axis=1) / np.maximum(norms, 1e-12)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    kept = []
    for index in np.flatnonzero(angle > settings.corner_angle_degrees)[
        np.argsort(angle[angle > settings.corner_angle_degrees])[::-1]
    ]:
        if kept:
            gap = np.abs(lengths[index] - lengths[np.array(kept)])
            if np.min(np.minimum(gap, perimeter - gap)) < window * 1.5:
                continue
        kept.append(index)
    return np.array(sorted(kept), dtype=np.int64)


def _side_line(points, lengths, low, high):
    chosen = np.flatnonzero((lengths >= low) & (lengths <= high))
    if len(chosen) < 3:
        return None
    side = points[chosen]
    centre = side.mean(axis=0)
    _, _, axes = np.linalg.svd(side - centre, full_matrices=False)
    direction = axes[0]
    spread = np.abs((side - centre) @ np.array([-direction[1], direction[0]]))
    return centre, direction, float(np.max(spread))


def _refine_corner(points, lengths, index, reach_back, reach_forward, settings):
    """True corner position: where the two sides would meet without antialiasing.

    Rendering shaves a sharp corner into a short chamfer, and the detector
    lands on one end of it. Intersecting the sides puts the anchor back.
    Returns (point, trim) where trim is how much arclength to drop each side.
    """
    trim = 1.5
    at = lengths[index]
    corner = points[index]
    span_back = min(4 * settings.corner_window, 0.45 * reach_back)
    span_forward = min(4 * settings.corner_window, 0.45 * reach_forward)
    if span_back <= trim + 1 or span_forward <= trim + 1:
        return corner, 0.0

    before = _side_line(points, lengths, at - span_back, at - trim)
    after = _side_line(points, lengths, at + trim, at + span_forward)
    if before is None or after is None:
        return corner, 0.0
    (point_a, direction_a, spread_a), (point_b, direction_b, spread_b) = before, after
    if max(spread_a, spread_b) > settings.tolerance:
        return corner, 0.0

    cross = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    if abs(cross) < np.sin(np.radians(20)):
        return corner, 0.0
    offset = point_b - point_a
    along = (offset[0] * direction_b[1] - offset[1] * direction_b[0]) / cross
    meeting = point_a + along * direction_a
    if np.linalg.norm(meeting - corner) > 2.5:
        return corner, 0.0

    # An antialiased sharp corner hugs its two sides; a corner the designer
    # rounded bulges off them. Sharpening that one would change the drawing.
    zone = points[np.flatnonzero(np.abs(lengths - at) <= trim)]
    if len(zone):
        normal_a = np.array([-direction_a[1], direction_a[0]])
        normal_b = np.array([-direction_b[1], direction_b[0]])
        off_a = np.abs((zone - meeting) @ normal_a)
        off_b = np.abs((zone - meeting) @ normal_b)
        if float(np.max(np.minimum(off_a, off_b))) > max(0.75, settings.tolerance):
            return None, 0.0
    return meeting, trim


# ---------------------------------------------------------------- batch fit


def _bernstein(parameters):
    inverse = 1.0 - parameters
    return inverse**3, 3 * inverse * inverse * parameters, 3 * inverse * parameters * parameters, parameters**3


def _evaluate(start, control_1, control_2, end, parameters):
    b0, b1, b2, b3 = _bernstein(parameters)
    return (
        b0[..., None] * start[:, None, :]
        + b1[..., None] * control_1[:, None, :]
        + b2[..., None] * control_2[:, None, :]
        + b3[..., None] * end[:, None, :]
    )


def _newton_step(start, control_1, control_2, end, samples, parameters):
    inverse = 1.0 - parameters
    point = _evaluate(start, control_1, control_2, end, parameters)
    s, c1, c2, e = start[:, None, :], control_1[:, None, :], control_2[:, None, :], end[:, None, :]
    first = (
        (3 * inverse * inverse)[..., None] * (c1 - s)
        + (6 * inverse * parameters)[..., None] * (c2 - c1)
        + (3 * parameters * parameters)[..., None] * (e - c2)
    )
    second = (6 * inverse)[..., None] * (c2 - 2 * c1 + s) + (6 * parameters)[..., None] * (e - 2 * c2 + c1)
    difference = point - samples
    numerator = np.sum(difference * first, axis=-1)
    denominator = np.sum(first * first, axis=-1) + np.sum(difference * second, axis=-1)
    safe = np.where(np.abs(denominator) > 1e-12, denominator, 1.0)
    step = np.where(np.abs(denominator) > 1e-12, numerator / safe, 0.0)
    return np.clip(parameters - step, 0.0, 1.0)


def _span_samples(points, firsts, lasts, count):
    fractions = np.linspace(0.0, 1.0, count)
    indices = np.rint(firsts[:, None] + fractions[None, :] * (lasts - firsts)[:, None]).astype(np.int64)
    return points[indices]


def fit_spans(points, firsts, lasts, tangents_out, tangents_in, free_start, free_end, settings):
    """Fit one curve per (first, last) span, all spans in one vectorized pass.

    tangents_out: (k, 2) unit tangents leaving each start; tangents_in:
    (k, 2) unit tangents pointing back into the curve from each end. A
    free end ignores its tangent and solves for its handle. Returns
    (max_error, squared_error, kinds, control_1, control_2).
    """
    firsts = np.asarray(firsts)
    lasts = np.asarray(lasts)
    count = len(lasts)
    samples = _span_samples(points, firsts, lasts, settings.samples)
    start = samples[:, 0, :]
    end = samples[:, -1, :]
    chord = end - start
    chord_length = np.maximum(np.hypot(chord[:, 0], chord[:, 1]), 1e-12)

    # The straight-line test uses a denser sample, so a bump can't hide
    # between points. It tolerates a few stray points: where two shapes
    # abut, renderers leak a little background into the seam, and one
    # wobble shouldn't split a perfectly straight edge.
    dense = _span_samples(points, firsts, lasts, settings.samples * 3) - start[:, None, :]
    distance = np.abs(dense[..., 0] * chord[:, None, 1] - dense[..., 1] * chord[:, None, 0]) / chord_length[:, None]
    ordered = np.sort(distance, axis=1)
    line_error = np.where(
        ordered[:, -1] <= 2.0 * settings.tolerance,
        ordered[:, int(0.97 * (ordered.shape[1] - 1))],
        np.inf,
    )

    steps = np.linalg.norm(np.diff(samples, axis=1), axis=-1)
    cumulative = np.concatenate((np.zeros((count, 1)), np.cumsum(steps, axis=1)), axis=1)
    parameters = cumulative / np.maximum(cumulative[:, -1:], 1e-12)

    limit = settings.max_handle_ratio * chord_length
    tangent_out = tangents_out
    control_1 = start + chord / 3
    control_2 = end - chord / 3

    for _ in range(settings.iterations + 1):
        b0, b1, b2, b3 = _bernstein(parameters)
        residual = samples - (b0 + b1)[..., None] * start[:, None, :] - (b2 + b3)[..., None] * end[:, None, :]

        columns = []
        if free_start:
            columns += [np.stack([b1, np.zeros_like(b1)], -1), np.stack([np.zeros_like(b1), b1], -1)]
        else:
            columns.append(b1[..., None] * tangent_out[:, None, :])
        if free_end:
            columns += [np.stack([b2, np.zeros_like(b2)], -1), np.stack([np.zeros_like(b2), b2], -1)]
        else:
            columns.append(b2[..., None] * tangents_in[:, None, :])

        design = np.stack([column.reshape(count, -1) for column in columns], axis=-1)
        target = residual.reshape(count, -1)
        normal = np.einsum("kmi,kmj->kij", design, design) + 1e-9 * np.eye(design.shape[-1])
        right = np.einsum("kmi,km->ki", design, target)
        solution = np.linalg.solve(normal, right[..., None])[..., 0]

        position = 0
        if free_start:
            handle_1 = solution[:, 0:2]
            position = 2
        else:
            handle_1 = solution[:, 0:1] * tangent_out
            position = 1
        if free_end:
            handle_2 = solution[:, position : position + 2]
        else:
            handle_2 = solution[:, position : position + 1] * tangents_in

        control_1, control_2 = _sanitise_handles(
            start, end, chord, chord_length, limit, handle_1, handle_2, tangent_out, tangents_in
        )
        parameters = _newton_step(start, control_1, control_2, end, samples, parameters)

    curve = _evaluate(start, control_1, control_2, end, parameters)
    error = np.linalg.norm(curve - samples, axis=-1)
    squared = np.sum(error * error, axis=1)
    max_error = np.max(error, axis=1)

    kinds = np.where(line_error <= settings.tolerance, "line", "cubic")
    is_line = kinds == "line"
    max_error = np.where(is_line, line_error, max_error)
    # The tie-break between equal-count paths must score a line as the line it is.
    line_squared = np.sum(distance * distance, axis=1) * (settings.samples / distance.shape[1])
    squared = np.where(is_line, line_squared, squared)
    control_1 = np.where(is_line[:, None], start + chord / 3, control_1)
    control_2 = np.where(is_line[:, None], end - chord / 3, control_2)
    return max_error, squared, kinds, control_1, control_2


def _dense_error(points, start, control_1, control_2, end, resolution=48):
    """Max distance from each span's dense points to its curve (as a fine polyline)."""
    steps = np.linspace(0.0, 1.0, resolution)
    polyline = _evaluate(start, control_1, control_2, end, np.broadcast_to(steps, (len(start), resolution)))
    first = polyline[:, :-1, :]
    span = polyline[:, 1:, :] - first
    length = np.maximum(np.sum(span * span, axis=-1), 1e-12)

    offset = points[:, :, None, :] - first[:, None, :, :]
    along = np.clip(np.sum(offset * span[:, None, :, :], axis=-1) / length[:, None, :], 0.0, 1.0)
    nearest = first[:, None, :, :] + along[..., None] * span[:, None, :, :]
    distance = np.min(np.linalg.norm(points[:, :, None, :] - nearest, axis=-1), axis=2)
    return np.max(distance, axis=1)


def _sanitise_handles(start, end, chord, chord_length, limit, handle_1, handle_2, tangent_out, tangents_in):
    """Reject handles that point backwards or overshoot; fall back to a third of the chord."""
    length_1 = np.hypot(handle_1[:, 0], handle_1[:, 1])
    length_2 = np.hypot(handle_2[:, 0], handle_2[:, 1])
    backwards_1 = np.sum(handle_1 * chord, axis=1) <= 1e-3 * chord_length * np.maximum(length_1, 1e-12)
    backwards_2 = np.sum(handle_2 * -chord, axis=1) <= 1e-3 * chord_length * np.maximum(length_2, 1e-12)

    fallback_1 = np.where(
        np.sum(tangent_out * chord, axis=1)[:, None] > 0, tangent_out, chord / chord_length[:, None]
    ) * (chord_length / 3)[:, None]
    fallback_2 = np.where(
        np.sum(tangents_in * -chord, axis=1)[:, None] > 0, tangents_in, -chord / chord_length[:, None]
    ) * (chord_length / 3)[:, None]

    handle_1 = np.where((backwards_1 | (length_1 < 1e-3 * chord_length))[:, None], fallback_1, handle_1)
    handle_2 = np.where((backwards_2 | (length_2 < 1e-3 * chord_length))[:, None], fallback_2, handle_2)

    for handle in (handle_1, handle_2):
        length = np.hypot(handle[:, 0], handle[:, 1])
        scale = np.where(length > limit, limit / np.maximum(length, 1e-12), 1.0)
        handle *= scale[:, None]
    return start + handle_1, end + handle_2


# ---------------------------------------------------------------- runs


@dataclass
class _Run:
    """One stretch of contour to fit: between two corners, or a whole smooth loop from one seam."""

    loop: int
    variant: int
    points: np.ndarray
    candidates: np.ndarray
    tangents: np.ndarray = None
    free_ends: bool = True
    offset: int = 0
    table: dict = None
    path: list = None
    segments: list = None


def _open_run(loop, points, settings):
    candidates = _douglas_peucker(points, settings.candidate_tolerance)
    run = _Run(loop, 0, points, candidates)
    if len(candidates) < 2:
        run.segments = []
    elif len(candidates) == 2:
        # Douglas–Peucker already proved every point lies near the chord.
        run.segments = _polygon(points, candidates)
    else:
        lengths, _ = _cumulative_length(points, closed=False)
        run.tangents = np.array(
            [_tangent(points, lengths, index, settings.tangent_window) for index in candidates]
        )
    return run


def _smooth_runs(loop, points, settings):
    """A loop with no corners, opened at each of its extrema — one run per seam."""
    _, perimeter = _cumulative_length(points, closed=True)
    runs = []
    for variant, seam in enumerate(_extrema_seams(points, settings.seams)):
        rolled = np.roll(points, -seam, axis=0)
        closed = np.vstack([rolled, rolled[:1]])
        rolled_lengths, _ = _cumulative_length(rolled, closed=True)
        candidates = _douglas_peucker(closed, settings.candidate_tolerance)
        run = _Run(loop, variant, closed, candidates, free_ends=False)
        if len(candidates) <= 2:
            run.segments = _polygon(closed, candidates)
            runs.append(run)
            continue

        pad = min(8, len(rolled) - 1)
        wrapped = np.vstack([rolled[len(rolled) - pad :], closed, rolled[1 : pad + 1]])
        wrapped_lengths = np.concatenate(
            (
                rolled_lengths[len(rolled) - pad :] - perimeter,
                np.append(rolled_lengths, perimeter),
                rolled_lengths[1 : pad + 1] + perimeter,
            )
        )
        tangents = np.array(
            [_tangent(wrapped, wrapped_lengths, index + pad, settings.tangent_window) for index in candidates]
        )
        tangents[-1] = tangents[0]
        run.tangents = tangents
        runs.append(run)
    return runs


def _prepare(loop, points, settings):
    """Split one closed loop into runs. Returns ("smooth" | "corners", runs)."""
    points = _dedupe(np.asarray(points, dtype=np.float64))
    if len(points) < 3:
        return "corners", []

    corners = find_corners(points, settings)
    if len(corners):
        points, corners, anchors = _settle_corners(points, corners, settings)
    if len(corners) == 0:
        return "smooth", _smooth_runs(loop, points, settings)

    lengths, perimeter = _cumulative_length(points, closed=True)
    at = lengths[corners]
    runs = []
    for index in range(len(corners)):
        start_point, start_trim = anchors[index]
        end_point, end_trim = anchors[(index + 1) % len(corners)]
        low = at[index] + start_trim
        high = (at[index + 1] if index + 1 < len(corners) else perimeter) - end_trim
        interior = points[(lengths > low) & (lengths < high)]
        runs.append(_open_run(loop, np.vstack([start_point, interior, end_point]), settings))
    return "corners", runs


# ---------------------------------------------------------------- batched fitting

_CHUNK = 4096


def _enumerate_spans(run, settings):
    """Candidate (first, last) pairs of one run that a single cubic could cover."""
    count = len(run.candidates)
    # One cubic can't turn through much more than a half-circle, so spans
    # whose tangent turns further are skipped without being fitted.
    headings = np.unwrap(np.arctan2(run.tangents[:, 1], run.tangents[:, 0]))
    turning = np.concatenate(([0.0], np.cumsum(np.abs(np.diff(headings)))))
    limit = np.radians(settings.max_turn_degrees)

    firsts, lasts = [], []
    for first in range(count - 1):
        stop = min(count, first + 1 + settings.max_span)
        reach = np.flatnonzero(turning[first + 1 : stop] - turning[first] <= limit)
        stop = first + 2 + (int(reach[-1]) if len(reach) else 0)
        firsts.extend([first] * (stop - first - 1))
        lasts.extend(range(first + 1, stop))
    return np.array(firsts, dtype=np.int64), np.array(lasts, dtype=np.int64)


def _build_tables(runs, settings):
    """Fit every candidate span of every run, a few vectorized batches for the whole image.

    Returns the concatenated point array the span indices refer to.
    """
    pending = [run for run in runs if run.segments is None]
    if not pending:
        return np.zeros((0, 2))

    offset = 0
    for run in pending:
        run.offset = offset
        run.table = {}
        offset += len(run.points)
    points = np.vstack([run.points for run in pending])

    parts = {name: [] for name in ("owner", "first", "last", "start_free", "end_free", "global_first", "global_last", "out", "in")}
    for index, run in enumerate(pending):
        first, last = _enumerate_spans(run, settings)
        count = len(run.candidates)
        parts["owner"].append(np.full(len(first), index, dtype=np.int64))
        parts["first"].append(first)
        parts["last"].append(last)
        parts["start_free"].append((first == 0) & run.free_ends)
        parts["end_free"].append((last == count - 1) & run.free_ends)
        parts["global_first"].append(run.offset + run.candidates[first])
        parts["global_last"].append(run.offset + run.candidates[last])
        parts["out"].append(run.tangents[first])
        parts["in"].append(-run.tangents[last])
    owners, firsts, lasts = (np.concatenate(parts[name]) for name in ("owner", "first", "last"))
    starts_free, ends_free = np.concatenate(parts["start_free"]), np.concatenate(parts["end_free"])
    global_first, global_last = np.concatenate(parts["global_first"]), np.concatenate(parts["global_last"])
    tangent_out, tangent_in = np.concatenate(parts["out"]), np.concatenate(parts["in"])

    scale = settings.tolerance * settings.tolerance
    for free_start in (False, True):
        for free_end in (False, True):
            group = np.flatnonzero((starts_free == free_start) & (ends_free == free_end))
            for start in range(0, len(group), _CHUNK):
                chunk = group[start : start + _CHUNK]
                worst, squared, kinds, control_1, control_2 = fit_spans(
                    points,
                    global_first[chunk],
                    global_last[chunk],
                    tangent_out[chunk],
                    tangent_in[chunk],
                    free_start,
                    free_end,
                    settings,
                )
                for row in np.flatnonzero(worst <= settings.tolerance):
                    span = chunk[row]
                    penalty = 1.0 + 1e-3 * squared[row] / (scale * settings.samples)
                    penalty -= 1e-4 if kinds[row] == "line" else 0.0
                    pending[owners[span]].table[(int(firsts[span]), int(lasts[span]))] = (
                        penalty,
                        kinds[row],
                        control_1[row],
                        control_2[row],
                    )
    return points


def _verify(checks, points, settings):
    """Dense check of chosen curves, all at once. Returns the (run, span) pairs that fail."""
    if not checks:
        return set()
    failed = set()
    for start in range(0, len(checks), _CHUNK):
        chunk = checks[start : start + _CHUNK]
        firsts = np.array([run.offset + run.candidates[span[0]] for run, span in chunk])
        lasts = np.array([run.offset + run.candidates[span[1]] for run, span in chunk])
        dense = _span_samples(points, firsts, lasts, settings.samples * 3)
        control_1 = np.array([run.table[span][2] for run, span in chunk])
        control_2 = np.array([run.table[span][3] for run, span in chunk])
        error = _dense_error(dense, dense[:, 0, :], control_1, control_2, dense[:, -1, :])
        failed.update((id(run), span) for (run, span), value in zip(chunk, error) if value > settings.tolerance)
    return failed


def _finish(run):
    segments = []
    for first, last in run.path:
        _, kind, control_1, control_2 = run.table[(first, last)]
        segments.append(
            Segment(kind, run.points[run.candidates[first]], control_1, control_2, run.points[run.candidates[last]])
        )
    run.segments = segments


def _solve(runs, points, settings):
    """Search every run, densely verify what was chosen, and search again where it failed.

    The fit only sees a sparse subsample, and a curve can cut a small
    corner between two samples and still look perfect. Verifying every
    span densely is expensive, so only chosen curves are checked.
    """
    active = [run for run in runs if run.segments is None]
    verified = set()
    for _ in range(12):
        if not active:
            return
        checks = []
        searched = []
        for run in active:
            run.path = _search(run.table, len(run.candidates), settings)
            if run.path is None:
                run.segments = _polygon(run.points, run.candidates)
                continue
            searched.append(run)
            for span in run.path:
                if run.table[span][1] == "cubic" and (id(run), span) not in verified:
                    checks.append((run, span))

        failed = _verify(checks, points, settings)
        verified.update((id(run), span) for run, span in checks if (id(run), span) not in failed)
        retry = set()
        for run, span in checks:
            if (id(run), span) in failed and span in run.table:
                del run.table[span]
                retry.add(id(run))
        for run in searched:
            if id(run) not in retry:
                _finish(run)
        active = [run for run in searched if id(run) in retry]

    # Out of rounds: take the best path over the spans that remain.
    for run in active:
        run.path = _search(run.table, len(run.candidates), settings)
        if run.path is None:
            run.segments = _polygon(run.points, run.candidates)
        else:
            _finish(run)


def _search(table, count, settings):
    """Breakpoint path through the span table, as a list of (first, last)."""
    outgoing = {}
    for (first, last), entry in table.items():
        outgoing.setdefault(first, []).append((last, entry[0]))

    if settings.greedy:
        path, node = [], 0
        while node < count - 1:
            reachable = [last for last, _ in outgoing.get(node, ())]
            if not reachable:
                return None
            path.append((node, max(reachable)))
            node = max(reachable)
        return path

    best = np.full(count, np.inf)
    best[0] = 0.0
    previous = np.full(count, -1, dtype=np.int64)
    for first in range(count - 1):
        if not np.isfinite(best[first]):
            continue
        for last, penalty in outgoing.get(first, ()):
            cost = best[first] + penalty
            if cost < best[last]:
                best[last] = cost
                previous[last] = first
    if not np.isfinite(best[-1]):
        return None

    path, node = [], count - 1
    while node > 0:
        path.append((int(previous[node]), node))
        node = int(previous[node])
    return path[::-1]



def _polygon(points, candidates):
    segments = []
    for first, last in zip(candidates[:-1], candidates[1:]):
        start, end = points[first], points[last]
        chord = end - start
        segments.append(Segment("line", start, start + chord / 3, end - chord / 3, end))
    return segments


def _extrema_seams(points, limit):
    """Leftmost, topmost, rightmost, bottommost points — where designers put anchors."""
    seams = []
    for index in (
        int(np.argmin(points[:, 0])),
        int(np.argmin(points[:, 1])),
        int(np.argmax(points[:, 0])),
        int(np.argmax(points[:, 1])),
    ):
        if all(np.linalg.norm(points[index] - points[other]) > 2.0 for other in seams):
            seams.append(index)
    return seams[:limit]


def _settle_corners(points, corners, settings):
    """Refine every corner; drop the ones that turn out to be rounded.

    Dropping one changes its neighbours' reach, so this repeats until the
    set is stable. Returns (rotated points, corner indices, anchors).
    """
    while True:
        points = np.roll(points, -int(corners[0]), axis=0)
        corners = (corners - corners[0]) % len(points)
        lengths, perimeter = _cumulative_length(points, closed=True)
        gaps = np.diff(np.append(lengths[corners], perimeter))
        unrolled = np.concatenate((lengths - perimeter, lengths, lengths + perimeter))
        tripled = np.vstack([points, points, points])
        anchors = [
            _refine_corner(tripled, unrolled, int(corner) + len(points), gaps[index - 1], gaps[index], settings)
            for index, corner in enumerate(corners)
        ]
        keep = [index for index, (point, _) in enumerate(anchors) if point is not None]
        if len(keep) == len(corners):
            return points, corners, anchors
        corners = corners[keep]
        if len(corners) == 0:
            return points, corners, []



def fit_loops(loops, settings):
    """Fit many closed loops together. Returns one list of Segments per loop.

    Every candidate span from every loop is fitted in shared vectorized
    batches, so the cost is a handful of large array operations rather
    than thousands of small ones.
    """
    prepared = [_prepare(index, points, settings) for index, points in enumerate(loops)]
    runs = [run for _, loop_runs in prepared for run in loop_runs]
    points = _build_tables(runs, settings)
    _solve(runs, points, settings)

    results = []
    for kind, loop_runs in prepared:
        if kind == "smooth":
            options = [run.segments for run in loop_runs if run.segments]
            results.append(min(options, key=len) if options else [])
        else:
            results.append([segment for run in loop_runs for segment in run.segments])
    return results


def fit_loop(points, settings):
    """Closed loop -> list of Segments, ending where it starts."""
    return fit_loops([points], settings)[0]
