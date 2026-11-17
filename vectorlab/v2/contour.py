"""Stage 2: sub-pixel contours recovered from antialiasing.

An antialiased edge pixel that is 30% red and 70% white records where
the edge crossed it. Quantising throws that away — v1 and VTracer both
trace integer pixel staircases and then try to smooth them back out.

Here each boundary pixel is unmixed between the colour inside the
region and the colour across the edge, giving its coverage. Marching
squares on that coverage field places the contour where the edge
actually was. Coverage is clamped either side of 0.5 so the contour's
topology is exactly the hard mask's — positions move, connectivity
never does.

Because coverage of A against B is one minus coverage of B against A,
two neighbouring regions trace the *same* curve along their shared edge.
"""

import numpy as np

_CLAMP = 0.02

# Unit-cell geometry, (x, y) with y down. Corner bits: tl=8 tr=4 br=2 bl=1.
_CORNERS = {"tl": (0.0, 0.0), "tr": (1.0, 0.0), "br": (1.0, 1.0), "bl": (0.0, 1.0)}
_EDGE_ENDS = {"T": ("tl", "tr"), "R": ("tr", "br"), "B": ("bl", "br"), "L": ("tl", "bl")}
_EDGE_MID = {"T": (0.5, 0.0), "R": (1.0, 0.5), "B": (0.5, 1.0), "L": (0.0, 0.5)}

# Saddles (5, 10) resolve as separated, matching 4-connected regions.
_CASE_SEGMENTS = {
    1: [("L", "B", ("bl",))],
    2: [("B", "R", ("br",))],
    3: [("L", "R", ("bl", "br"))],
    4: [("T", "R", ("tr",))],
    5: [("T", "R", ("tr",)), ("L", "B", ("bl",))],
    6: [("T", "B", ("tr", "br"))],
    7: [("T", "L", ("tr", "br", "bl"))],
    8: [("T", "L", ("tl",))],
    9: [("T", "B", ("tl", "bl"))],
    10: [("T", "L", ("tl",)), ("B", "R", ("br",))],
    11: [("T", "R", ("tl", "bl", "br"))],
    12: [("L", "R", ("tl", "tr"))],
    13: [("R", "B", ("tl", "tr", "bl"))],
    14: [("L", "B", ("tl", "tr", "br"))],
}


def _orient(first, second, inside_corners):
    """Order an edge pair so the inside lies on the walker's left."""
    start = np.array(_EDGE_MID[first])
    end = np.array(_EDGE_MID[second])
    direction = end - start
    left = np.array([direction[1], -direction[0]])
    centre = np.mean([_CORNERS[name] for name in inside_corners], axis=0)
    if np.dot(centre - (start + end) / 2.0, left) > 0:
        return first, second
    return second, first


_ORIENTED = {
    case: [_orient(first, second, corners) for first, second, corners in segments]
    for case, segments in _CASE_SEGMENTS.items()
}


def _unmix(colour, inside, outside):
    """Fraction of `inside` in a two-colour blend, and the blend's residual."""
    span = inside - outside
    reach = np.sum(span * span, axis=-1)
    mix = np.sum((colour - outside) * span, axis=-1) / np.maximum(reach, 1e-9)
    mix = np.clip(mix, 0.0, 1.0)
    residual = np.linalg.norm(colour - (outside + mix[..., None] * span), axis=-1)
    residual = np.where(reach < 1e-6, np.inf, residual)
    return mix, residual


def coverage_field(inside, labels, colours, palette, report=False):
    """Coverage of the masked region at every pixel of a crop.

    inside:  (h, w) bool — the region, already cropped with a margin
    labels:  (h, w) int  — palette index per pixel, -1 outside the image
    colours: (h, w, 3) float in [0, 1]
    palette: (k, 3) float in [0, 1]

    With report=True also returns the mask of pixels whose coverage was
    measured from a colour edge, as opposed to left hard.
    """
    field = inside.astype(np.float64)
    in_image = labels >= 0
    padded_inside = np.pad(inside, 1, constant_values=False)
    padded_labels = np.pad(labels, 1, constant_values=-1)

    neighbours = []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rows = slice(1 + dy, padded_inside.shape[0] - 1 + dy)
        columns = slice(1 + dx, padded_inside.shape[1] - 1 + dx)
        neighbours.append((padded_inside[rows, columns], padded_labels[rows, columns]))

    own_colour = palette[np.clip(labels, 0, None)]
    best_in = np.full(inside.shape, np.inf)
    mix_in = np.ones(inside.shape)
    best_out = np.full(inside.shape, np.inf)
    mix_out = np.zeros(inside.shape)

    for neighbour_inside, neighbour_label in neighbours:
        valid = neighbour_label >= 0
        neighbour_colour = palette[np.clip(neighbour_label, 0, None)]

        # Inside pixel facing outward: blend of its own colour and the neighbour's.
        facing_out = inside & ~neighbour_inside & valid & in_image
        mix, residual = _unmix(colours, own_colour, neighbour_colour)
        better = facing_out & (residual < best_in)
        best_in[better] = residual[better]
        mix_in[better] = mix[better]

        # Outside pixel facing inward: blend of the neighbour's colour and its own.
        facing_in = ~inside & neighbour_inside & valid & in_image
        mix, residual = _unmix(colours, neighbour_colour, own_colour)
        better = facing_in & (residual < best_out)
        best_out[better] = residual[better]
        mix_out[better] = mix[better]

    inner = np.isfinite(best_in)
    outer = np.isfinite(best_out)
    field[inner] = np.clip(mix_in[inner], 0.5 + _CLAMP, 1.0)
    field[outer] = np.clip(mix_out[outer], 0.0, 0.5 - _CLAMP)
    if report:
        return field, inner | outer
    return field


def _crossing(value_in, value_out):
    """Distance from the inside sample to the edge, assuming a box-filtered step."""
    full_in = value_in >= 1.0 - 1e-6
    empty_out = value_out <= 1e-6
    if full_in and empty_out:
        return 0.5
    if empty_out:
        return value_in - 0.5
    if full_in:
        return 0.5 + value_out
    return (value_in - 0.5) / (value_in - value_out)


def trace_loops(field):
    """Closed, consistently oriented iso-0.5 loops of a field padded with zeros.

    Points are in sample coordinates: sample (row, column) sits at
    (x=column, y=row).
    """
    inside = field > 0.5
    width = field.shape[1]
    case = (
        (inside[:-1, :-1].astype(np.int8) << 3)
        | (inside[:-1, 1:].astype(np.int8) << 2)
        | (inside[1:, 1:].astype(np.int8) << 1)
        | inside[1:, :-1].astype(np.int8)
    )
    horizontal_offset = 0
    vertical_offset = field.shape[0] * width

    successor = {}
    position = {}

    def edge_point(row, column, edge):
        corner_a, corner_b = _EDGE_ENDS[edge]
        offsets = {"tl": (0, 0), "tr": (0, 1), "br": (1, 1), "bl": (1, 0)}
        ra, ca = row + offsets[corner_a][0], column + offsets[corner_a][1]
        rb, cb = row + offsets[corner_b][0], column + offsets[corner_b][1]
        value_a = field[ra, ca]
        value_b = field[rb, cb]
        if value_a > 0.5:
            distance = _crossing(value_a, value_b)
            start, end = (ca, ra), (cb, rb)
        else:
            distance = _crossing(value_b, value_a)
            start, end = (cb, rb), (ca, ra)
        return (start[0] + distance * (end[0] - start[0]), start[1] + distance * (end[1] - start[1]))

    def edge_key(row, column, edge):
        if edge == "T":
            return horizontal_offset + row * width + column
        if edge == "B":
            return horizontal_offset + (row + 1) * width + column
        if edge == "L":
            return vertical_offset + row * width + column
        return vertical_offset + row * width + column + 1

    for row, column in np.argwhere((case != 0) & (case != 15)):
        for first, second in _ORIENTED[int(case[row, column])]:
            source = edge_key(row, column, first)
            target = edge_key(row, column, second)
            successor[source] = target
            if source not in position:
                position[source] = edge_point(row, column, first)
            if target not in position:
                position[target] = edge_point(row, column, second)

    loops = []
    while successor:
        start, following = successor.popitem()
        points = [position[start]]
        current = following
        while current != start:
            points.append(position[current])
            current = successor.pop(current, None)
            if current is None:
                points = []
                break
        if len(points) >= 3:
            loops.append(np.array(points, dtype=np.float64))
    return loops


def signed_area(loop):
    x = loop[:, 0]
    y = loop[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
