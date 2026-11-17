"""Stage 4a: recognising the shapes designers actually draw.

A tracer approximates a circle with a ring of cubics and a rounded
rectangle with eight segments. A designer drew one `<circle>` and one
`<rect rx>`. When a contour fits one of these within tolerance, emit the
primitive: fewer anchors, and the result edits the way it was made.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Primitive:
    kind: str  # "circle", "ellipse", "rect"
    centre: tuple
    size: tuple  # circle: (r,), ellipse: (rx, ry), rect: (width, height)
    angle: float = 0.0  # degrees, ellipse only
    radius: float = 0.0  # rect corner radius


def _quantile(values, fraction):
    ordered = np.sort(values)
    return float(ordered[int(fraction * (len(ordered) - 1))])


def _accepts(residual, tolerance):
    """Robust acceptance: almost every point close, none far."""
    return _quantile(residual, 0.98) <= tolerance and float(np.max(residual)) <= 1.6 * tolerance


def _circle(points, tolerance):
    x, y = points[:, 0], points[:, 1]
    design = np.stack([x, y, np.ones_like(x)], axis=1)
    target = -(x * x + y * y)
    (d, e, f), *_ = np.linalg.lstsq(design, target, rcond=None)
    centre = np.array([-d / 2.0, -e / 2.0])
    squared = centre @ centre - f
    if squared <= 4.0:
        return None
    radius = float(np.sqrt(squared))
    residual = np.abs(np.hypot(x - centre[0], y - centre[1]) - radius)
    if not _accepts(residual, tolerance):
        return None
    return Primitive("circle", (float(centre[0]), float(centre[1])), (radius,))


def _ellipse(points, tolerance):
    """Halíř–Flusser direct ellipse fit, judged by first-order geometric distance."""
    centre_guess = points.mean(axis=0)
    scale = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1e-9) / 2.0
    x = (points[:, 0] - centre_guess[0]) / scale
    y = (points[:, 1] - centre_guess[1]) / scale

    quadratic = np.stack([x * x, x * y, y * y], axis=1)
    linear = np.stack([x, y, np.ones_like(x)], axis=1)
    s1 = quadratic.T @ quadratic
    s2 = quadratic.T @ linear
    s3 = linear.T @ linear
    try:
        transform = -np.linalg.solve(s3, s2.T)
    except np.linalg.LinAlgError:
        return None
    reduced = s1 + s2 @ transform
    constraint_inverse = np.array([[0, 0, 0.5], [0, -1, 0], [0.5, 0, 0]])
    values, vectors = np.linalg.eig(constraint_inverse @ reduced)
    vectors = np.real(vectors)
    condition = 4 * vectors[0] * vectors[2] - vectors[1] ** 2
    positive = np.flatnonzero(condition > 0)
    if not len(positive):
        return None
    first = vectors[:, positive[0]]
    a, b, c = first
    d, e, f = transform @ first

    determinant = b * b - 4 * a * c
    if determinant >= 0:
        return None
    centre_x = (2 * c * d - b * e) / determinant
    centre_y = (2 * a * e - b * d) / determinant
    numerator = 2 * (a * e * e + c * d * d - b * d * e + determinant * f)
    root = np.sqrt((a - c) ** 2 + b * b)
    axis_1 = -np.sqrt(abs(numerator * (a + c + root))) / determinant
    axis_2 = -np.sqrt(abs(numerator * (a + c - root))) / determinant
    angle = 0.5 * np.arctan2(-b, c - a)

    conic = a * x * x + b * x * y + c * y * y + d * x + e * y + f
    gradient = np.hypot(2 * a * x + b * y + d, b * x + 2 * c * y + e)
    residual = np.abs(conic) / np.maximum(gradient, 1e-12) * scale
    if not _accepts(residual, tolerance):
        return None

    radius_x, radius_y = float(abs(axis_1) * scale), float(abs(axis_2) * scale)
    degrees = float(np.degrees(angle))
    centre = (float(centre_x * scale + centre_guess[0]), float(centre_y * scale + centre_guess[1]))
    if radius_x < radius_y:
        radius_x, radius_y = radius_y, radius_x
        degrees += 90.0
    degrees = ((degrees + 90.0) % 180.0) - 90.0
    if abs(degrees) < 0.5:
        degrees = 0.0
    return Primitive("ellipse", centre, (radius_x, radius_y), angle=degrees)


def _rounded_box_distance(points, centre, half, radius):
    offset = np.abs(points - centre) - (half - radius)
    outside = np.hypot(np.maximum(offset[:, 0], 0), np.maximum(offset[:, 1], 0))
    inside = np.minimum(np.maximum(offset[:, 0], offset[:, 1]), 0)
    return np.abs(outside + inside - radius)


def _rect(points, tolerance):
    """Axis-aligned rectangle, with the corner radius that fits best."""
    low = points.min(axis=0)
    high = points.max(axis=0)
    half = (high - low) / 2.0
    if np.min(half) < 1.5:
        return None
    centre = (low + high) / 2.0

    # Any rounded rect fills at least π/4 of its box; skip the search otherwise.
    x, y = points[:, 0], points[:, 1]
    area = 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
    if area < 0.76 * 4 * half[0] * half[1]:
        return None

    # Antialiasing softens any small corner by about a pixel, so a sharp
    # rectangle often fits marginally worse than a slightly rounded one.
    # If sharp fits within tolerance, sharp is what was drawn.
    sharp = _rounded_box_distance(points, centre, half, 0.0)
    if _accepts(sharp, tolerance):
        return Primitive(
            "rect", (float(centre[0]), float(centre[1])), (float(2 * half[0]), float(2 * half[1]))
        )

    best_radius, best_error = 0.0, np.inf
    left, right = 0.0, float(np.min(half))
    for _ in range(24):
        first = left + (right - left) * 0.382
        second = left + (right - left) * 0.618
        error_first = _quantile(_rounded_box_distance(points, centre, half, first), 0.98)
        error_second = _quantile(_rounded_box_distance(points, centre, half, second), 0.98)
        if error_first < error_second:
            right = second
        else:
            left = first
    for radius in (0.0, (left + right) / 2.0):
        error = _quantile(_rounded_box_distance(points, centre, half, radius), 0.98)
        if error < best_error:
            best_radius, best_error = radius, error

    if best_radius < 0.75:
        best_radius = 0.0
    residual = _rounded_box_distance(points, centre, half, best_radius)
    if not _accepts(residual, tolerance):
        return None
    return Primitive(
        "rect",
        (float(centre[0]), float(centre[1])),
        (float(2 * half[0]), float(2 * half[1])),
        radius=float(best_radius),
    )


def detect_primitive(points, tolerance):
    """The simplest primitive that fits the loop, or None."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 12:
        return None
    # Circle before rect: a square with full corner radius is also a circle,
    # and the designer meant the circle.
    for detector in (_circle, _rect, _ellipse):
        found = detector(points, tolerance)
        if found is not None:
            return found
    return None
