"""Strokes: a thin shape of even width becomes a centreline with a stroke width.

Traced as a filled shape, a 3 px line is two parallel outlines that
must be edited together. A designer drew one line. A shape is a stroke
when a disc of one fixed width, swept along a centreline, reproduces it.

The centreline comes from thinning the shape's mask, then each point is
moved to halfway between the two sides of the sub-pixel outline, which
also measures the width there.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

_MAX_WIDTH = 14.0  # px: wider than this is a shape, not a line
_MIN_WIDTH = 1.0  # px: thinner than this is an antialiasing fringe
_MIN_LENGTH_RATIO = 4.0  # a line is at least this many widths long
_EVEN_SHARE = 0.8  # share of centreline points whose width must be near the median
_EVEN_LOW, _EVEN_HIGH = 0.7, 1.35
_EVEN_DEPTH_SHARE = 0.6  # a first, looser test of evenness, before the outline is traced
_MAX_MISMATCH = 0.6  # px of disagreement with the mask per px of centreline...
_MAX_MISMATCH_SHARE = 0.25  # ...and at most this share of the width, so thin lines are held as tightly
_SUPERSAMPLE = 4

_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))


@dataclass
class Stroke:
    chains: list  # open centrelines, (n, 2) image coordinates
    loops: list  # closed centrelines
    width: float  # image pixels
    cap: str = "round"  # "round" (round joins) or "butt" (mitred joins)


# ---------------------------------------------------------------- centreline


def _depth(mask, width):
    """How many erosions each pixel survives, counting to half `width` and beyond: twice a
    centre pixel's depth, less one, is roughly the line's width there."""
    depth = mask.astype(np.int16)
    eroded = mask.copy()
    for _ in range(int(np.ceil(width / 2.0))):
        padded = np.pad(eroded, 1, constant_values=False)
        eroded = (
            eroded
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
        if not eroded.any():
            break
        depth += eroded
    return depth


def _too_wide(mask, width):
    """True if some part of the mask survives eroding by half the width."""
    return bool(_depth(mask, width).max() > np.ceil(width / 2.0))


def _thin(mask):
    """Zhang–Suen thinning to a one-pixel skeleton."""
    image = np.pad(mask, 1).astype(np.uint8)
    while True:
        changed = False
        for phase in (0, 1):
            ring = [image[1 + dy : image.shape[0] - 1 + dy, 1 + dx : image.shape[1] - 1 + dx] for dy, dx in _OFFSETS[1:] + _OFFSETS[:1]]
            # ring order: N, NE, E, SE, S, SW, W, NW
            north, _, east, _, south, _, west, _ = ring
            count = sum(ring)
            transitions = sum(((ring[i] == 0) & (ring[(i + 1) % 8] == 1)).astype(np.uint8) for i in range(8))
            if phase == 0:
                guard = (north * east * south == 0) & (east * south * west == 0)
            else:
                guard = (north * east * west == 0) & (north * south * west == 0)
            remove = (image[1:-1, 1:-1] == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & guard
            if remove.any():
                image[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            return image[1:-1, 1:-1].astype(bool)


def _graph(skeleton):
    """Neighbour lists of skeleton pixels. A diagonal link is dropped where a
    two-step straight path joins the same pixels, so staircases aren't triangles."""
    pixels = {(int(row), int(column)) for row, column in zip(*np.nonzero(skeleton))}
    links = {pixel: [] for pixel in pixels}
    for row, column in pixels:
        for dy, dx in _OFFSETS:
            other = (row + dy, column + dx)
            if other not in pixels:
                continue
            if dy and dx and ((row + dy, column) in pixels or (row, column + dx) in pixels):
                continue
            links[(row, column)].append(other)
    return links


def _chains(links):
    """Split a skeleton into chains between junctions and ends, plus closed cycles."""
    nodes = {pixel for pixel, near in links.items() if len(near) != 2}
    seen = set()
    chains, cycles = [], []
    for node in nodes:
        for first in links[node]:
            if (node, first) in seen:
                continue
            chain, previous, current = [node], node, first
            seen.add((node, first))
            while True:
                chain.append(current)
                if current in nodes:
                    seen.add((current, previous))
                    break
                following = [pixel for pixel in links[current] if pixel != previous]
                if not following:
                    break
                previous, current = current, following[0]
                seen.add((previous, current))
            chains.append((chain, len(links[node]) == 1, len(links[chain[-1]]) == 1))
    visited = {pixel for chain, _, _ in chains for pixel in chain}
    for start in links:
        if start in visited or len(links[start]) != 2:
            continue
        cycle, previous, current = [start], None, start
        while True:
            following = [pixel for pixel in links[current] if pixel != previous and pixel not in cycle[1:]]
            if not following or following[0] == start:
                break
            previous, current = current, following[0]
            cycle.append(current)
        visited.update(cycle)
        if len(cycle) >= 4:
            cycles.append(cycle)
    return chains, cycles


def _prune(chains, width):
    """Drop short branches that end in the open: thinning leaves them at corners and ends."""
    return [
        (chain, free_start, free_end)
        for chain, free_start, free_end in chains
        if not ((free_start or free_end) and not (free_start and free_end) and len(chain) < 1.5 * width)
    ]


# ---------------------------------------------------------------- sub-pixel centre


def _normals(points, closed):
    count = len(points)
    if closed:
        ahead, behind = np.roll(points, -2, axis=0), np.roll(points, 2, axis=0)
    else:
        index = np.arange(count)
        ahead, behind = points[np.minimum(index + 2, count - 1)], points[np.maximum(index - 2, 0)]
    tangent = ahead - behind
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    return np.column_stack([-tangent[:, 1], tangent[:, 0]])


def _centre_on_outline(points, normals, segments, reach):
    """Move each point to midway between the outline on either side; also return the width there."""
    starts, ends = segments
    span = ends - starts
    centred, widths = points.copy(), np.full(len(points), np.nan)
    for first in range(0, len(points), 256):
        p, n = points[first : first + 256, None, :], normals[first : first + 256, None, :]
        denominator = n[..., 0] * span[None, :, 1] - n[..., 1] * span[None, :, 0]
        offset = starts[None] - p
        along = (offset[..., 0] * span[None, :, 1] - offset[..., 1] * span[None, :, 0]) / np.where(
            np.abs(denominator) < 1e-12, np.nan, denominator
        )
        where = (offset[..., 0] * n[..., 1] - offset[..., 1] * n[..., 0]) / np.where(
            np.abs(denominator) < 1e-12, np.nan, denominator
        )
        hits = (where >= 0) & (where <= 1) & (np.abs(along) <= reach)
        ahead = np.where(hits & (along > 0), along, np.inf).min(axis=1)
        behind = np.where(hits & (along <= 0), along, -np.inf).max(axis=1)
        found = np.isfinite(ahead) & np.isfinite(behind)
        rows = np.arange(first, first + len(p))[found]
        centred[rows] = points[rows] + ((ahead[found] + behind[found]) / 2.0)[:, None] * normals[rows]
        widths[rows] = ahead[found] - behind[found]
    return centred, widths


# ---------------------------------------------------------------- check


def _dense(polyline, closed, spacing=0.5):
    points = np.vstack([polyline, polyline[:1]]) if closed else polyline
    out = [points[:1]]
    for start, end in zip(points[:-1], points[1:]):
        steps = max(1, int(np.ceil(np.linalg.norm(end - start) / spacing)))
        out.append(start + (end - start) * (np.arange(1, steps + 1) / steps)[:, None])
    return np.vstack(out)


def _mismatch(stroke, mask, origin):
    """Pixels where the stroke as drawn and the mask disagree, per pixel of centreline.

    The stroke is drawn as a disc swept along the centreline; a butt cap
    then cuts off what lies beyond each open end.
    """
    height, width = mask.shape
    canvas = Image.new("L", (width * _SUPERSAMPLE, height * _SUPERSAMPLE), 0)
    draw = ImageDraw.Draw(canvas)
    radius = stroke.width / 2.0 * _SUPERSAMPLE
    length = 0.0
    for polyline, closed in [(chain, False) for chain in stroke.chains] + [(loop, True) for loop in stroke.loops]:
        points = (np.vstack([polyline, polyline[:1]]) if closed else polyline) - origin
        length += float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        scaled = [tuple(point) for point in points * _SUPERSAMPLE]
        # A wide line with rounded joins, and a disc at each vertex for the ends and sharp turns.
        if len(scaled) > 1:
            draw.line(scaled, fill=255, width=max(1, round(2 * radius)), joint="curve")
        for x, y in scaled:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    if stroke.cap == "butt":
        for chain in stroke.chains:
            for at_start in (True, False):
                draw.polygon([tuple(point) for point in _beyond_end(chain - origin, at_start, stroke.width) * _SUPERSAMPLE], fill=0)
    coverage = np.array(canvas, dtype=np.float64).reshape(height, _SUPERSAMPLE, width, _SUPERSAMPLE).mean(axis=(1, 3)) / 255.0
    return float(((coverage >= 0.5) != mask).sum()) / max(length, 1.0)


def _beyond_end(chain, at_start, width):
    """The square past one end of a chain, which a butt cap leaves unpainted."""
    end, inner = (chain[0], chain[1]) if at_start else (chain[-1], chain[-2])
    direction = end - inner
    direction /= max(np.linalg.norm(direction), 1e-12)
    across = np.array([-direction[1], direction[0]]) * (width / 2.0 + 1.0)
    ahead = direction * (width / 2.0 + 1.0)
    return np.array([end + across, end + across + ahead, end - across + ahead, end - across])


# ---------------------------------------------------------------- detection


def detect_stroke(mask, origin, outline):
    """A Stroke if the masked shape is one even-width line, else None.

    mask: (h, w) bool crop of the shape, holes not filled.
    origin: image coordinates of the crop's top-left corner.
    outline: a function giving the shape's sub-pixel boundary loops, in image coordinates.
    Tracing it costs more than everything before it, so it's only done for shapes the
    skeleton alone doesn't rule out.
    """
    depth = _depth(mask, _MAX_WIDTH)
    if mask.sum() < 12 or depth.max() > np.ceil(_MAX_WIDTH / 2.0):
        return None
    skeleton_mask = _thin(mask)
    # An even line has about the same depth all along its middle; a blob of texture doesn't.
    along = depth[skeleton_mask]
    if len(along) and np.mean(np.abs(along - np.median(along)) <= 1) < _EVEN_DEPTH_SHARE:
        return None
    chains, cycles = _chains(_graph(skeleton_mask))
    skeleton = sum(len(chain) for chain, _, _ in chains) + sum(len(cycle) for cycle in cycles)
    estimate = mask.sum() / max(1, skeleton)
    chains = _prune(chains, estimate)
    if not chains and not cycles:
        return None
    # Area over skeleton length is close to the width. Margins keep these checks looser
    # than the measured ones that follow.
    if not 0.6 * _MIN_WIDTH <= estimate <= 1.3 * _MAX_WIDTH or skeleton < 0.8 * _MIN_LENGTH_RATIO * estimate:
        return None
    outline = outline()
    if not outline:
        return None

    starts = np.vstack([loop for loop in outline])
    ends = np.vstack([np.roll(loop, -1, axis=0) for loop in outline])
    centre = np.array([0.5, 0.5]) + origin

    def refine(pixels, closed):
        points = np.array(pixels, dtype=np.float64)[:, ::-1] + centre
        return _centre_on_outline(points, _normals(points, closed), (starts, ends), _MAX_WIDTH)

    refined = [(refine(chain, False), free_start, free_end) for chain, free_start, free_end in chains]
    rounds = [refine(cycle, True) for cycle in cycles]

    widths = np.concatenate([w for (_, w), _, _ in refined] + [w for _, w in rounds])
    widths = widths[np.isfinite(widths)]
    if len(widths) < 4:
        return None
    width = float(np.median(widths))
    even = float(np.mean((widths >= _EVEN_LOW * width) & (widths <= _EVEN_HIGH * width)))
    length = sum(len(chain) for chain, _, _ in chains) + sum(len(cycle) for cycle in cycles)
    if not _MIN_WIDTH <= width <= _MAX_WIDTH or even < _EVEN_SHARE or length < _MIN_LENGTH_RATIO * width:
        return None

    lines = [
        _Line(_settle(points, w, width), _junction_key(chain[0], free_start), _junction_key(chain[-1], free_end))
        for ((points, w), free_start, free_end), (chain, _, _) in zip(refined, chains)
    ]
    lines = _place_junctions(lines, width)
    rings = [_settle(points, w, width) for points, w in rounds]
    candidates = []
    for cap, inset in (("round", width / 2.0), ("butt", 0.0)):
        capped = [_Line(_extend_tips(line, inset, width, (starts, ends)), line.start, line.end) for line in lines]
        chains_out, closed = _join_straight(capped, width)
        stroke = Stroke(chains=chains_out, loops=rings + closed, width=width, cap=cap)
        candidates.append((_mismatch(stroke, mask, origin), stroke))
        if not chains_out:
            break  # closed lines have no caps to choose between
    # On a thin line the two caps differ by a pixel or two, within noise; round is
    # what line art usually uses, so butt has to be clearly better.
    mismatch, stroke = candidates[0]
    if len(candidates) > 1 and candidates[1][0] < min(0.85 * mismatch, mismatch - 0.05):
        mismatch, stroke = candidates[1]
    return stroke if mismatch <= min(_MAX_MISMATCH, _MAX_MISMATCH_SHARE * width) else None


def could_be_stroke(mask):
    """Cheap first test: big enough, and no part wider than a line can be."""
    return mask.sum() >= 12 and not _too_wide(mask, _MAX_WIDTH)


def _settle(points, widths, width):
    """Drop points whose measured width is far off: near a junction the normal
    crosses another branch. Unmeasured points and the two ends stay."""
    keep = ~np.isfinite(widths) | (widths <= 2.0 * width)
    keep[0] = keep[-1] = True
    return points[keep]


# ---------------------------------------------------------------- junctions and tips


@dataclass
class _Line:
    points: np.ndarray
    start: object  # junction key, or None for a free tip
    end: object


def _junction_key(pixel, free):
    return None if free else pixel


def _end_line(points, at_start, width):
    """(end point, unit direction pointing out of the line) from the points near one end,
    skipping the first stretch, where a junction or tip distorts them."""
    ordered = points if at_start else points[::-1]
    steps = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(ordered, axis=0), axis=1))))
    chosen = ordered[(steps >= 0.75 * width) & (steps <= 0.75 * width + max(4.0, 2.0 * width))]
    if len(chosen) < 2:
        chosen = ordered[: max(2, min(len(ordered), 4))]
    centre = chosen.mean(axis=0)
    _, _, axes = np.linalg.svd(chosen - centre, full_matrices=False)
    direction = axes[0]
    if (ordered[0] - centre) @ direction < 0:
        direction = -direction
    return centre, direction


def _clusters(keys, links):
    """Junction pixels within two pixels of each other, or joined by a link, are one junction."""
    keys = sorted(keys)
    parent = {key: key for key in keys}

    def find(key):
        while parent[key] != key:
            key = parent[key]
        return key

    for index, first in enumerate(keys):
        for second in keys[index + 1 :]:
            if abs(first[0] - second[0]) <= 2 and abs(first[1] - second[1]) <= 2:
                parent[find(second)] = find(first)
    for first, second in links:
        parent[find(second)] = find(first)
    return {key: find(key) for key in keys}


def _length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _place_junctions(lines, width):
    """Move every line end at a junction to where the lines meeting there intersect.

    Thinning splits one crossing of several lines into a few junctions a
    stroke-width apart, joined by short stubs. The stubs are dropped and
    their junctions treated as one.
    """
    stubs = [
        line for line in lines
        if line.start is not None and line.end is not None and line.start != line.end and _length(line.points) < 3.0 * width
    ]
    lines = [line for line in lines if not any(line is stub for stub in stubs)]
    keys = {key for line in lines + stubs for key in (line.start, line.end) if key is not None}
    cluster = _clusters(keys, [(stub.start, stub.end) for stub in stubs])
    for line in lines:
        line.start = cluster.get(line.start) if line.start is not None else None
        line.end = cluster.get(line.end) if line.end is not None else None

    meeting = {}
    for index, line in enumerate(lines):
        for at_start in (True, False):
            key = line.start if at_start else line.end
            if key is not None and len(line.points) >= 2:
                meeting.setdefault(key, []).append((index, at_start, _end_line(line.points, at_start, width)))

    for key, ends in meeting.items():
        point = _intersection([end_line for _, _, end_line in ends])
        for index, at_start, _ in ends:
            lines[index].points = _attach(lines[index].points, at_start, point, width)
    return lines


def _intersection(end_lines):
    """Least-squares meeting point of lines given as (point, direction)."""
    normal_matrix = np.zeros((2, 2))
    right = np.zeros(2)
    for point, direction in end_lines:
        normal = np.array([-direction[1], direction[0]])
        normal_matrix += np.outer(normal, normal)
        right += normal * (normal @ point)
    eigenvalues = np.linalg.eigvalsh(normal_matrix)
    if eigenvalues[0] < 0.05 * eigenvalues[-1]:
        # Nearly parallel lines don't pin a point; take the middle of their ends.
        return np.mean([point for point, _ in end_lines], axis=0)
    return np.linalg.solve(normal_matrix, right)


def _attach(points, at_start, point, width):
    """Replace the end of a line with `point`, dropping the distorted stretch next to it."""
    ordered = points if at_start else points[::-1]
    distance = np.linalg.norm(ordered - point, axis=1)
    keep = np.flatnonzero(distance > 1.5 * width)
    rest = ordered[keep[0] :] if len(keep) else ordered[-1:]
    joined = np.vstack([point, rest])
    return joined if at_start else joined[::-1]


def _extend_tips(line, inset, width, segments):
    """Move each free tip to `inset` short of the outline: half a width for a
    round cap, whose centre sits there, and none for a butt cap."""
    points = line.points
    for at_start, free in ((True, line.start is None), (False, line.end is None)):
        if not free or len(points) < 2:
            continue
        centre, direction = _end_line(points, at_start, width)
        end = points[0] if at_start else points[-1]
        base = centre + ((end - centre) @ direction) * direction
        reach = _ray_hit(base, direction, segments)
        if reach is None:
            continue
        tip = base + max(0.0, reach - inset) * direction
        trimmed = points[1:] if at_start else points[:-1]
        points = np.vstack([tip, trimmed]) if at_start else np.vstack([trimmed, tip])
    return points


def _ray_hit(origin, direction, segments):
    starts, ends = segments
    span = ends - starts
    denominator = direction[0] * span[:, 1] - direction[1] * span[:, 0]
    safe = np.where(np.abs(denominator) < 1e-12, np.nan, denominator)
    offset = starts - origin
    along = (offset[:, 0] * span[:, 1] - offset[:, 1] * span[:, 0]) / safe
    where = (offset[:, 0] * direction[1] - offset[:, 1] * direction[0]) / safe
    hits = along[(where >= 0) & (where <= 1) & (along > 0)]
    return float(hits.min()) if len(hits) else None


def _join_straight(lines, width, max_turn=30.0):
    """Join lines that run straight through a junction into one, as a designer drew them.

    Returns (open chains, closed loops).
    """
    lines = [_Line(line.points, line.start, line.end) for line in lines]
    closed = []
    limit = -np.cos(np.radians(max_turn))
    while True:
        best = None
        ends = {}
        for index, line in enumerate(lines):
            for at_start in (True, False):
                key = line.start if at_start else line.end
                if key is not None and len(line.points) >= 2:
                    ends.setdefault(key, []).append((index, at_start, _end_line(line.points, at_start, width)[1]))
        for key, meeting in ends.items():
            for first in range(len(meeting)):
                for second in range(first + 1, len(meeting)):
                    alignment = float(meeting[first][2] @ meeting[second][2])
                    if alignment < limit and (best is None or alignment < best[0]):
                        best = (alignment, meeting[first], meeting[second])
        if best is None:
            break
        (index_a, start_a, _), (index_b, start_b, _) = best[1], best[2]
        a, b = lines[index_a], lines[index_b]
        if index_a == index_b:
            if len(a.points) >= 5:
                closed.append(a.points[:-1])
            lines = [line for position, line in enumerate(lines) if position != index_a]
            continue
        first_points = a.points[::-1] if start_a else a.points
        first_far = a.end if start_a else a.start
        second_points = b.points if start_b else b.points[::-1]
        second_far = b.end if start_b else b.start
        merged = _Line(np.vstack([first_points, second_points[1:]]), first_far, second_far)
        lines = [line for position, line in enumerate(lines) if position not in (index_a, index_b)] + [merged]

    return [line.points for line in lines], closed
