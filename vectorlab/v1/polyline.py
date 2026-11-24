"""Stage 4: path simplification.

Douglas-Peucker on a closed ring. Epsilon is in pixels; ~1.0 removes
the staircase without eating real detail.
"""

import numpy as np


def _perpendicular_distances(points, start, end):
    line = end - start
    length = float(np.hypot(line[0], line[1]))
    if length < 1e-12:
        return np.hypot(points[:, 0] - start[0], points[:, 1] - start[1])
    normal = np.array([-line[1], line[0]]) / length
    return np.abs((points - start) @ normal)


def _douglas_peucker(points, epsilon):
    if len(points) < 3:
        return points

    keep = np.zeros(len(points), dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack = [(0, len(points) - 1)]

    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        interior = points[first + 1 : last]
        distances = _perpendicular_distances(interior, points[first], points[last])
        worst = int(np.argmax(distances))
        if distances[worst] > epsilon:
            split = first + 1 + worst
            keep[split] = True
            stack.append((first, split))
            stack.append((split, last))

    return points[keep]


def simplify_ring(ring, epsilon):
    """Simplify a closed ring, anchored on its extreme point.

    Anchoring on a guaranteed-corner vertex stops the open-curve
    algorithm from cutting across the seam.
    """
    if len(ring) < 4:
        return ring

    anchor = int(np.lexsort((ring[:, 1], ring[:, 0]))[0])
    rolled = np.roll(ring, -anchor, axis=0)
    closed = np.vstack([rolled, rolled[:1]])
    simplified = _douglas_peucker(closed, epsilon)

    if len(simplified) > 1 and np.allclose(simplified[0], simplified[-1]):
        simplified = simplified[:-1]

    return simplified


def turn_angles(ring, window=1):
    """Turn angle in radians at each vertex of a closed ring.

    `window` sets the scale the angle is measured over: 1 compares the
    two adjacent edges, larger values compare the chords reaching
    `window` vertices back and forward. Measuring at scale 1 on a noisy
    boundary flags nearly every vertex as a corner, because quantisation
    wiggle turns sharply at the single-edge scale while the contour it
    belongs to is smooth. A window of 2-3 separates real corners from
    that noise.
    """
    span = max(1, int(window))
    previous = np.roll(ring, span, axis=0)
    following = np.roll(ring, -span, axis=0)
    incoming = ring - previous
    outgoing = following - ring

    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    dot = incoming[:, 0] * outgoing[:, 0] + incoming[:, 1] * outgoing[:, 1]
    return np.abs(np.arctan2(cross, dot))
