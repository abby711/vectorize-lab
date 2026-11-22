"""Stage 1: palette reduction.

Maps every pixel to one of N palette entries: full colour by k-means,
grayscale by luminance bands, or black and white by a threshold.
"""

import numpy as np

_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)

_D65_WHITE = np.array([0.95047, 1.00000, 1.08883])


def srgb_to_lab(rgb):
    """rgb: float array in [0,1], shape (..., 3) -> CIE Lab."""
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _SRGB_TO_XYZ.T / _D65_WHITE

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)

    lab = np.empty_like(xyz)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def squared_distances(points, centres):
    """(n, k) squared distances, by ||p||^2 - 2 p.c + ||c||^2: a matrix product instead of an
    (n, k, 3) difference array."""
    return np.maximum(
        (points * points).sum(axis=1)[:, None] - 2.0 * points @ centres.T + (centres * centres).sum(axis=1)[None, :], 0.0
    )


def _kmeans_plus_plus_init(points, k, rng):
    centers = [points[rng.integers(len(points))]]
    closest = np.sum((points - centers[0]) ** 2, axis=1)

    for _ in range(k - 1):
        total = float(closest.sum())
        if total <= 0.0:
            centers.append(points[rng.integers(len(points))])
            continue
        pick = rng.random() * total
        index = int(np.searchsorted(np.cumsum(closest), pick))
        index = min(index, len(points) - 1)
        centers.append(points[index])
        closest = np.minimum(closest, np.sum((points - centers[-1]) ** 2, axis=1))

    return np.array(centers)


def _kmeans(points, k, iterations, rng):
    if len(points) <= k:
        return points.copy(), np.arange(len(points))

    centers = _kmeans_plus_plus_init(points, k, rng)
    labels = np.zeros(len(points), dtype=np.int32)

    for _ in range(iterations):
        # Chunked distance computation keeps peak memory bounded.
        for start in range(0, len(points), 65536):
            labels[start : start + 65536] = np.argmin(squared_distances(points[start : start + 65536], centers), axis=1)

        # Every cluster's mean at once; an empty cluster keeps its centre.
        sizes = np.bincount(labels, minlength=k)
        sums = np.stack([np.bincount(labels, weights=points[:, c], minlength=k) for c in range(points.shape[1])], axis=1)
        occupied = sizes > 0
        means = sums[occupied] / sizes[occupied][:, None]
        moved = not np.allclose(means, centers[occupied], atol=1e-4)
        centers[occupied] = means

        if not moved:
            break

    return centers, labels


def quantize_full_color(rgb, color_count, sample_limit=40000, seed=7):
    """rgb: uint8 (h, w, 3). Returns (label_map int32, palette uint8 (k,3))."""
    height, width = rgb.shape[:2]
    flat = rgb.reshape(-1, 3)
    # Each distinct colour is converted and assigned once; pixels look theirs up.
    keys = (flat[:, 0].astype(np.int32) << 16) | (flat[:, 1].astype(np.int32) << 8) | flat[:, 2]
    unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    colours = np.stack([(unique_keys >> 16) & 255, (unique_keys >> 8) & 255, unique_keys & 255], axis=1) / 255.0
    colour_lab = srgb_to_lab(colours)

    rng = np.random.default_rng(seed)
    if len(flat) > sample_limit:
        sample_index = rng.choice(len(flat), sample_limit, replace=False)
        sample = colour_lab[inverse[sample_index]]
    else:
        sample = colour_lab[inverse]

    centers, _ = _kmeans(sample, color_count, iterations=24, rng=rng)

    colour_labels = np.zeros(len(colour_lab), dtype=np.int32)
    for start in range(0, len(colour_lab), 65536):
        colour_labels[start : start + 65536] = np.argmin(squared_distances(colour_lab[start : start + 65536], centers), axis=1)
    labels = colour_labels[inverse]

    # Empty clusters are dropped and the labels renumbered, so the palette
    # never reports a colour that no pixel actually uses.
    used = np.unique(labels)
    remap = np.full(len(centers), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)

    # Palette in sRGB: mean of the actual pixels assigned to each center,
    # which reads truer than converting the Lab centroid back.
    members = np.bincount(colour_labels, weights=counts, minlength=len(centers))
    sums = np.stack(
        [np.bincount(colour_labels, weights=colours[:, c] * counts, minlength=len(centers)) for c in range(3)], axis=1
    )
    means = sums[used] / members[used][:, None]
    palette = np.clip(means * 255.0, 0, 255).astype(np.uint8)

    return remap[labels].reshape(height, width), palette


def quantize_grayscale(rgb, gray_count):
    """Uniform luminance banding. Returns (label_map, palette)."""
    luma = (
        0.2126 * rgb[..., 0].astype(np.float64)
        + 0.7152 * rgb[..., 1].astype(np.float64)
        + 0.0722 * rgb[..., 2].astype(np.float64)
    )
    bands = max(2, gray_count)
    labels = np.clip((luma / 256.0 * bands).astype(np.int32), 0, bands - 1)

    palette = np.zeros((bands, 3), dtype=np.uint8)
    for index in range(bands):
        members = luma[labels == index]
        level = int(round(members.mean())) if members.size else int(255 * index / (bands - 1))
        palette[index] = (level, level, level)

    return labels, palette


def quantize_monochrome(rgb, threshold):
    """Two-tone split at `threshold` (0-255). Returns (label_map, palette)."""
    luma = (
        0.2126 * rgb[..., 0].astype(np.float64)
        + 0.7152 * rgb[..., 1].astype(np.float64)
        + 0.0722 * rgb[..., 2].astype(np.float64)
    )
    labels = (luma >= float(threshold)).astype(np.int32)
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    return labels, palette
