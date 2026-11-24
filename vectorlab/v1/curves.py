"""Stage 5+6: corner detection and cubic Bezier fitting.

This is the stage that decides whether output looks designed or looks
auto-traced. Vertices whose turn angle exceeds `corner_threshold` stay
sharp; everything else gets a Catmull-Rom tangent, converted to cubic
Bezier control points.
"""

import numpy as np

from .polyline import turn_angles


def _hermite_to_bezier(point_a, point_b, tangent_a, tangent_b):
    control_1 = point_a + tangent_a / 3.0
    control_2 = point_b - tangent_b / 3.0
    return control_1, control_2


def fit_ring(ring, corner_threshold, smoothness, corner_window=2):
    """Fit a closed ring. Returns a list of segment dicts.

    Each segment is {"kind": "line"|"cubic", ...} in absolute
    coordinates, ready for SVG emission.
    """
    count = len(ring)
    if count < 3:
        return []

    window = min(corner_window, max(1, count // 4))
    angles = turn_angles(ring, window)
    is_corner = angles > corner_threshold

    previous = np.roll(ring, 1, axis=0)
    following = np.roll(ring, -1, axis=0)
    tangents = smoothness * (following - previous) * 0.5
    tangents[is_corner] = 0.0

    segments = []
    for index in range(count):
        next_index = (index + 1) % count
        point_a = ring[index]
        point_b = ring[next_index]
        tangent_a = tangents[index]
        tangent_b = tangents[next_index]

        straight = float(np.hypot(*tangent_a)) < 1e-9 and float(np.hypot(*tangent_b)) < 1e-9
        if straight:
            segments.append({"kind": "line", "to": point_b})
            continue

        control_1, control_2 = _hermite_to_bezier(point_a, point_b, tangent_a, tangent_b)
        segments.append({"kind": "cubic", "c1": control_1, "c2": control_2, "to": point_b})

    return segments


def polygon_segments(ring):
    """Straight-line fallback, useful for eyeballing the trace stage."""
    return [{"kind": "line", "to": ring[(i + 1) % len(ring)]} for i in range(len(ring))]
