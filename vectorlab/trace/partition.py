"""Stage 1: palette, labelling, and a partition where nothing is dropped.

Three ideas, each fixing a v1 failure measured on the benchmark:

- The palette is *detected*, not guessed. Flat artwork has a handful of
  exact colours, each covering thousands of pixels; antialiasing blends
  are spread thinly across many values. Histogram peaks find the real
  palette, so nobody has to pick k.
- Edge pixels are labelled as *blends*. A pixel between blue and yellow
  can sit nearer to green in colour space; v1 called it green and grew a
  sliver. Here it is explained as a mix of its neighbours' colours.
- Undersized regions are *merged*, never deleted, into the neighbour
  they share the most boundary with. Every pixel stays painted.
"""

from dataclasses import dataclass

import numpy as np

from ..core.colour import quantize_full_color, squared_distances, srgb_to_lab
from ..core.regions import lab_image, label_partition, unique_colours

_BIN_SHIFT = 3
_BINS_PER_CHANNEL = 256 >> _BIN_SHIFT


@dataclass
class Partition:
    palette: np.ndarray  # (k, 3) uint8 sRGB
    label_map: np.ndarray  # (h, w) int32 palette index per pixel
    component_map: np.ndarray  # (h, w) int32 component id per pixel
    component_label: np.ndarray  # (n,) palette index per component
    areas: np.ndarray  # (n,) pixel count per component
    boxes: np.ndarray  # (n, 4) y0, x0, y1, x1 exclusive
    palette_mode: str  # "detected" or "kmeans"


def _box_sum_3d(volume):
    """Sum over each bin's 3×3×3 neighbourhood."""
    padded = np.pad(volume, [(1, 1)] * 3 + [(0, 0)] * (volume.ndim - 3))
    total = np.zeros_like(volume)
    size = volume.shape[0]
    for dz in range(3):
        for dy in range(3):
            for dx in range(3):
                total += padded[dz : dz + size, dy : dy + size, dx : dx + size]
    return total


def _histogram_peaks(rgb, min_fraction):
    """Local maxima of the colour histogram. Returns (colours, masses)."""
    flat = rgb.reshape(-1, 3)
    bins = (flat >> _BIN_SHIFT).astype(np.int64)
    side = _BINS_PER_CHANNEL
    keys = (bins[:, 0] * side + bins[:, 1]) * side + bins[:, 2]

    counts = np.bincount(keys, minlength=side**3).reshape(side, side, side).astype(np.float64)
    sums = np.stack(
        [np.bincount(keys, weights=flat[:, channel], minlength=side**3) for channel in range(3)],
        axis=1,
    ).reshape(side, side, side, 3)

    # A colour straddling a bin edge splits across two bins whose means are
    # nearly identical; two distinct colours in neighbouring bins are not.
    # So candidates are every well-populated bin, each centred on its own
    # pixels, and near-identical means are merged later in Lab.
    neighbourhood_counts = _box_sum_3d(counts)
    min_count = max(12, min_fraction * len(flat))
    candidate = (counts > 0) & (neighbourhood_counts >= min_count)
    peak_index = np.argwhere(candidate)
    masses = counts[candidate]
    colours = sums[candidate] / masses[:, None]

    order = np.argsort(masses)[::-1]
    return colours[order], masses[order], peak_index[order]


def _interior_counts(nearest, pure, palette_size, reach=1):
    """Per colour: pure pixels with the same pure colour up to `reach` away on all four sides."""
    solid = np.pad(np.where(pure, nearest, -1), reach, constant_values=-1)
    height, width = nearest.shape
    centre = solid[reach : reach + height, reach : reach + width]
    interior = centre >= 0
    for step in range(1, reach + 1):
        for dy, dx in ((-step, 0), (step, 0), (0, -step), (0, step)):
            interior &= solid[reach + dy : reach + dy + height, reach + dx : reach + dx + width] == centre
    return np.bincount(centre[interior], minlength=palette_size)


def _blend_residual(colour, palette):
    """How far `colour` is from the nearest two-colour mix of `palette` entries."""
    best = np.inf
    for first in range(len(palette)):
        for second in range(first + 1, len(palette)):
            span = palette[first] - palette[second]
            reach = float(span @ span)
            if reach < 1e-9:
                continue
            mix = np.clip(float((colour - palette[second]) @ span) / reach, 0.0, 1.0)
            if 0.08 < mix < 0.92:
                best = min(best, float(np.linalg.norm(colour - (palette[second] + mix * span))))
    return best


def _keep_solid_colours(candidates, interior, members, min_interior, pixel_count, max_colors):
    """Greedy by interior size; colours that could be blends must prove a real interior."""
    floats = candidates.astype(np.float64) / 255.0
    order = np.argsort(interior)[::-1]
    kept = []
    for index in order:
        if interior[index] < min_interior or len(kept) >= max_colors:
            continue
        if len(kept) >= 2 and _blend_residual(floats[index], floats[kept]) < 0.05:
            strong = interior[index] >= max(40, 0.0005 * pixel_count)
            solid = interior[index] >= 0.3 * max(members[index], 1)
            if not (strong and solid):
                continue
        kept.append(index)
    return np.array(kept, dtype=np.int64)


def denoise_texture(rgb, raw_noise):
    """Median-filter textured input before labelling; leave clean artwork alone.

    Spray texture, grain and fine detail otherwise become hundreds of
    thousands of speckle regions. A median filter removes speckle while
    keeping edges where they are, and clean artwork never reaches it.
    """
    if raw_noise < 1.5:
        return rgb
    from PIL import Image, ImageFilter

    size = 3 if raw_noise < 3.0 else 5
    return np.array(Image.fromarray(rgb).filter(ImageFilter.MedianFilter(size)))


def estimate_noise(rgb):
    """Mean neighbour ΔE inside flat areas: 0 for clean artwork, ~0.5 for JPEG.

    Edges are excluded, so this measures noise rather than detail. It sets
    how close two colours may be and still count as different: 4 ΔE apart
    is two tints in a clean PNG, and just compression noise in a JPEG.
    """
    lab = lab_image(rgb)
    centre = lab[1:-1, 1:-1]
    differences = np.stack(
        [
            np.linalg.norm(lab[1:-1, 2:] - centre, axis=-1),
            np.linalg.norm(lab[1:-1, :-2] - centre, axis=-1),
            np.linalg.norm(lab[2:, 1:-1] - centre, axis=-1),
            np.linalg.norm(lab[:-2, 1:-1] - centre, axis=-1),
        ]
    )
    flat = differences.max(axis=0) < 10.0
    if flat.mean() < 0.05:
        # Almost nothing is flat: this isn't artwork with noise on it, it
        # is mostly noise, and the typical neighbour difference says how much.
        return float(np.median(differences.mean(axis=0)))
    return float(differences.mean(axis=0)[flat].mean())


def detect_palette(rgb, max_colors, min_fraction, merge_delta, pure_delta, noise=0.0, reach=1):
    """Histogram-peak palette, kept only where colours form solid regions.

    Returns (palette uint8, coverage fraction of pixels explained purely).
    """
    colours, masses, _ = _histogram_peaks(rgb, min_fraction)
    if not len(colours):
        return np.zeros((0, 3), dtype=np.uint8), 0.0

    distinct = 2.5 + 12.0 * noise
    accepted_rgb = []
    accepted_lab = []
    accepted_mass = []
    for colour, mass in zip(colours[:128], masses[:128]):
        lab = srgb_to_lab(colour[None, :] / 255.0)[0]
        if accepted_lab and _is_shoulder(
            lab, mass, accepted_lab, accepted_mass, max(merge_delta, 1.5 * distinct), distinct
        ):
            continue
        accepted_rgb.append(colour)
        accepted_lab.append(lab)
        accepted_mass.append(mass)

    candidates = np.clip(np.round(np.array(accepted_rgb)), 0, 255).astype(np.uint8)
    nearest, distance = _nearest_palette(rgb, candidates)
    pure = distance <= pure_delta

    # A real colour has an interior. Antialiasing blends and compression
    # ringing live in bands one or two pixels wide and never do.
    interior = _interior_counts(nearest, pure, len(candidates), reach)
    members = np.bincount(nearest.ravel(), minlength=len(candidates))
    pixel_count = rgb.shape[0] * rgb.shape[1]
    min_interior = max(6, int(0.00005 * pixel_count))
    keep = _keep_solid_colours(candidates, interior, members, min_interior, pixel_count, max_colors)
    if len(keep) < 1:
        return np.zeros((0, 3), dtype=np.uint8), 0.0

    palette = candidates[np.sort(keep)]
    nearest, distance = _nearest_palette(rgb, palette)
    pure = distance <= pure_delta
    flat = rgb.reshape(-1, 3)
    for index in range(len(palette)):
        members = flat[(nearest.ravel() == index) & pure.ravel()]
        if len(members):
            palette[index] = np.clip(np.round(members.mean(axis=0)), 0, 255).astype(np.uint8)

    return palette, float(pure.mean())


def _is_shoulder(lab, count, accepted_lab, accepted_count, merge_delta, distinct=2.5):
    """True when a histogram peak is noise around a colour already accepted.

    Nearness alone can't decide it: a white card on a near-white page is
    5.9 ΔE apart and both are real. A real colour has mass of its own;
    compression noise around a colour is a thin shoulder of its peak.
    """
    distances = np.linalg.norm(np.array(accepted_lab) - lab, axis=1)
    if np.min(distances) < distinct:
        return True
    close = distances < merge_delta
    if not close.any():
        return False
    return bool(count < 0.25 * np.max(np.array(accepted_count)[close]))


def _nearest_palette(rgb, palette):
    """Nearest palette entry in Lab. Returns (index map, ΔE map)."""
    height, width = rgb.shape[:2]
    colours, inverse = unique_colours(rgb)
    lab = srgb_to_lab(colours.astype(np.float64) / 255.0)
    palette_lab = srgb_to_lab(palette.astype(np.float64) / 255.0)

    index = np.zeros(len(lab), dtype=np.int32)
    distance = np.zeros(len(lab), dtype=np.float64)
    for start in range(0, len(lab), 65536):
        chunk = lab[start : start + 65536]
        squared = np.sum((chunk[:, None, :] - palette_lab[None, :, :]) ** 2, axis=2)
        index[start : start + 65536] = np.argmin(squared, axis=1)
        distance[start : start + 65536] = np.sqrt(np.min(squared, axis=1))

    return index[inverse].reshape(height, width), distance[inverse].reshape(height, width)


def assign_labels(rgb, palette, pure_delta, blend_residual, report=False, reach=1):
    """Nearest-colour labels, with edge pixels explained as two-colour blends.

    `reach` is how far a blend can extend from an edge: 1 pixel for
    antialiasing, about 3 for JPEG, whose half-resolution colour smears
    every edge. Candidate colours come from pixels up to that far away.

    With report=True also returns the mask of pixels explained as blends.
    """
    labels, distance = _nearest_palette(rgb, palette)
    impure = distance > pure_delta
    if not impure.any() or len(palette) < 2:
        return (labels, np.zeros(labels.shape, dtype=bool)) if report else labels

    padded = np.pad(labels, reach, mode="edge")
    ys, xs = np.nonzero(impure)
    columns = [labels[ys, xs]]
    for step in range(1, reach + 1):
        for dy, dx in ((-step, 0), (step, 0), (0, -step), (0, step)):
            columns.append(padded[ys + reach + dy, xs + reach + dx])
    candidates = np.stack(columns, axis=1)

    colours = rgb[ys, xs].astype(np.float64) / 255.0
    palette_float = palette.astype(np.float64) / 255.0
    best_residual = np.full(len(ys), np.inf)
    best_label = labels[ys, xs].copy()

    width = candidates.shape[1]
    for first in range(width):
        for second in range(first + 1, width):
            label_a = candidates[:, first]
            label_b = candidates[:, second]
            colour_a = palette_float[label_a]
            colour_b = palette_float[label_b]
            span = colour_a - colour_b
            reach = np.sum(span * span, axis=1)
            mix = np.clip(np.sum((colours - colour_b) * span, axis=1) / np.maximum(reach, 1e-9), 0, 1)
            residual = np.linalg.norm(colours - (colour_b + mix[:, None] * span), axis=1)
            residual[label_a == label_b] = np.inf

            better = residual < best_residual
            best_residual[better] = residual[better]
            best_label[better] = np.where(mix[better] >= 0.5, label_a[better], label_b[better])

    explained = best_residual <= blend_residual
    labels[ys[explained], xs[explained]] = best_label[explained]
    if report:
        blends = np.zeros(labels.shape, dtype=bool)
        blends[ys[explained], xs[explained]] = True
        return labels, blends
    return labels


def refine_palette(rgb, palette, labels, blends, options, pure_delta, rounds=2, reach=1):
    """Add colours that solid groups of pixels need but the palette lacks.

    A thin highlight or a small field of colour covers too little of the
    image to show up as a histogram peak, so its pixels get the nearest
    palette colour even when that is 30 ΔE away. Every pixel should be
    explained by its colour, directly or as a blend of its neighbours;
    where clusters of pixels aren't, the palette is missing a colour.
    """
    lab = lab_image(rgb)
    for _ in range(rounds):
        palette_lab = srgb_to_lab(palette.astype(np.float64) / 255.0)
        error = np.linalg.norm(lab - palette_lab[labels], axis=-1)
        unexplained = (error > options.residual_delta) & ~blends

        # Keep pixels whose unexplained neighbours look like them: a real
        # colour forms runs and patches, isolated misfits are noise.
        padded = np.pad(unexplained, 1)
        padded_lab = np.pad(lab, ((1, 1), (1, 1), (0, 0)), mode="edge")
        support = np.zeros(unexplained.shape, dtype=np.int8)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbour = padded[1 + dy : padded.shape[0] - 1 + dy, 1 + dx : padded.shape[1] - 1 + dx]
            neighbour_lab = padded_lab[1 + dy : padded_lab.shape[0] - 1 + dy, 1 + dx : padded_lab.shape[1] - 1 + dx]
            support += (neighbour & (np.linalg.norm(neighbour_lab - lab, axis=-1) < 8.0)).astype(np.int8)
        solid = unexplained & (support >= (2 if reach == 1 else 4))
        if solid.sum() < options.min_missing_colour:
            return palette, labels, blends

        found = _missing_colours(rgb[solid], palette_lab, options)
        if not len(found):
            return palette, labels, blends
        palette = np.concatenate([palette, found])
        labels, blends = assign_labels(rgb, palette, pure_delta, options.blend_residual, report=True, reach=reach)
    return palette, labels, blends


def _missing_colours(pixels, palette_lab, options, limit=24):
    """Distinct colours among unexplained pixels, most common first."""
    bins = (pixels >> _BIN_SHIFT).astype(np.int64)
    side = _BINS_PER_CHANNEL
    keys = (bins[:, 0] * side + bins[:, 1]) * side + bins[:, 2]
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    accepted, accepted_lab = [], []
    for index in np.argsort(counts)[::-1]:
        if counts[index] < options.min_missing_colour or len(accepted) >= limit:
            break
        colour = pixels[inverse == index].mean(axis=0)
        lab = srgb_to_lab(colour[None, :] / 255.0)[0]
        known = np.concatenate([palette_lab, np.array(accepted_lab).reshape(-1, 3)])
        if np.min(np.linalg.norm(known - lab, axis=1)) < options.residual_delta * 0.75:
            continue
        accepted.append(colour)
        accepted_lab.append(lab)
    return np.clip(np.round(np.array(accepted).reshape(-1, 3)), 0, 255).astype(np.uint8)


def _adjacency(component_map):
    """Shared-boundary counts between touching components, both directions."""
    pairs = []
    for left, right in (
        (component_map[:, :-1], component_map[:, 1:]),
        (component_map[:-1, :], component_map[1:, :]),
    ):
        differs = left != right
        pairs.append(np.stack([left[differs], right[differs]], axis=1))
        pairs.append(np.stack([right[differs], left[differs]], axis=1))
    edges = np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=np.int64)
    if not len(edges):
        return edges, np.zeros(0, dtype=np.int64)
    unique, counts = np.unique(edges.astype(np.int64), axis=0, return_counts=True)
    return unique, counts


def merge_small_regions(label_map, min_area, image_lab=None, palette_lab=None, floor_area=None, keep_contrast=None, image_rgb=None, palette_rgb=None, sliver_reach=0, sliver_cap=400):
    """Absorbs undersized components into a neighbour.

    One greedy pass over the region adjacency graph, smallest first, with
    the graph updated as regions merge — exact, and far cheaper than
    relabelling the image round after round.

    Without colours, a region goes to the neighbour it shares the most
    boundary with. With colours, it goes to the neighbour whose colour is
    closest to its own — a brown speck bordered mostly by a green tree is
    still brown — and a small region that stands out sharply from every
    neighbour is kept, down to `floor_area`: speckle is low-contrast, a
    window or a stamen tip is not.
    """
    import heapq

    component_map, component_label = label_partition(label_map)
    areas = np.bincount(component_map.ravel()).astype(np.int64)
    if len(areas) < 2 or (areas.min() >= min_area and sliver_reach <= 1):
        return label_map

    edges, counts = _adjacency(component_map)
    neighbours = [dict() for _ in range(len(areas))]
    for (first, second), count in zip(edges.tolist(), counts.tolist()):
        neighbours[first][second] = count

    colour_aware = image_lab is not None and palette_lab is not None
    if colour_aware:
        flat = component_map.ravel()
        means = np.stack(
            [np.bincount(flat, weights=image_lab[..., channel].ravel(), minlength=len(areas)) for channel in range(3)],
            axis=1,
        ) / np.maximum(areas, 1)[:, None]
        if image_rgb is not None and palette_rgb is not None:
            rgb_means = np.stack(
                [np.bincount(flat, weights=image_rgb[..., channel].ravel(), minlength=len(areas)) for channel in range(3)],
                axis=1,
            ) / np.maximum(areas, 1)[:, None] / 255.0
            palette_float = palette_rgb.astype(np.float64) / 255.0
        else:
            rgb_means = None
    floor = min_area if floor_area is None else min(floor_area, min_area)

    root = np.arange(len(areas))
    heap = [(int(area), int(node)) for node, area in enumerate(areas) if area < min_area]
    heapq.heapify(heap)

    def absorb(node, target):
        area = int(areas[node])
        if colour_aware:
            means[target] = (means[target] * areas[target] + means[node] * area) / (areas[target] + area)
            if rgb_means is not None:
                rgb_means[target] = (rgb_means[target] * areas[target] + rgb_means[node] * area) / (areas[target] + area)
        for other, count in neighbours[node].items():
            del neighbours[other][node]
            if other != target:
                neighbours[target][other] = neighbours[target].get(other, 0) + count
                neighbours[other][target] = neighbours[other].get(target, 0) + count
        neighbours[node] = {}
        root[node] = target
        areas[target] += area
        if areas[target] < min_area:
            heapq.heappush(heap, (int(areas[target]), int(target)))
    while heap:
        area, node = heapq.heappop(heap)
        if root[node] != node or area != areas[node] or area >= min_area or not neighbours[node]:
            continue
        if colour_aware:
            others = list(neighbours[node].items())
            distance = np.linalg.norm(
                palette_lab[component_label[[other for other, _ in others]]] - means[node], axis=1
            )
            blend_target = None
            if keep_contrast is not None and area >= floor and float(distance.min()) >= keep_contrast:
                blend_target = _blend_owner(rgb_means, palette_float, component_label, node, others) if rgb_means is not None else None
                if blend_target is None:
                    continue
            if blend_target is not None:
                target = blend_target
            else:
                order = sorted(range(len(others)), key=lambda index: (round(float(distance[index]), 1), -others[index][1]))
                target = others[order[0]][0]
        else:
            target = max(neighbours[node].items(), key=lambda item: (item[1], -item[0]))[0]
        absorb(node, target)

    # JPEG smears colour across edges into thin bands that can be large in
    # area. A band no wider than the smear whose colour is a mix of the two
    # sides is a fringe, whatever its area.
    if sliver_reach > 1 and colour_aware and rgb_means is not None:
        for area, node in sorted((int(areas[n]), n) for n in range(len(areas)) if root[n] == n and areas[n] <= sliver_cap):
            if root[node] != node or area != areas[node] or not neighbours[node]:
                continue
            boundary = sum(neighbours[node].values())
            if boundary == 0 or area / boundary > 0.6 * sliver_reach:
                continue
            owner = _blend_owner(rgb_means, palette_float, component_label, node, list(neighbours[node].items()))
            if owner is not None:
                absorb(node, owner)

        # Averaging removes noise. A real region averages to its own colour;
        # a patch that per-pixel noise pushed across a close colour boundary
        # averages to the neighbour's. That patch was mislabelled.
        for _ in range(2):
            moved = False
            for area, node in sorted((int(areas[n]), n) for n in range(len(areas)) if root[n] == n and areas[n] <= 20 * sliver_cap):
                if root[node] != node or not neighbours[node]:
                    continue
                own = float(np.linalg.norm(means[node] - palette_lab[component_label[node]]))
                others = list(neighbours[node])
                distance = np.linalg.norm(palette_lab[component_label[others]] - means[node], axis=1)
                closest = int(np.argmin(distance))
                if float(distance[closest]) + 0.5 < own:
                    absorb(node, others[closest])
                    moved = True
            if not moved:
                break

    # Follow merges to their final owner.
    for node in range(len(root)):
        owner = node
        while root[owner] != owner:
            owner = root[owner]
        root[node] = owner
    return component_label[root][component_map].astype(np.int32)


def _blend_owner(rgb_means, palette_float, component_label, node, others, tolerance=0.05):
    """If a small region's colour is a mix of two neighbours' colours, the neighbour that dominates.

    A JPEG fringe between orange and white can land exactly on light
    yellow and look like a real light-yellow speck. A window can't be made
    by mixing its facade with the sky. Only the first is merged.
    """
    colour = rgb_means[node]
    best, owner = tolerance, None
    neighbour_ids = [other for other, _ in others]
    for first in range(len(neighbour_ids)):
        for second in range(first + 1, len(neighbour_ids)):
            colour_a = palette_float[component_label[neighbour_ids[first]]]
            colour_b = palette_float[component_label[neighbour_ids[second]]]
            span = colour_a - colour_b
            reach = float(span @ span)
            if reach < 1e-6:
                continue
            mix = float((colour - colour_b) @ span) / reach
            if not 0.1 < mix < 0.9:
                continue
            residual = float(np.linalg.norm(colour - (colour_b + mix * span)))
            if residual < best:
                best = residual
                owner = neighbour_ids[first] if mix >= 0.5 else neighbour_ids[second]
    return owner


def _merge_small_regions_rounds(label_map, min_area, rounds=40):
    """Round-based merging, kept as a reference for the graph version."""
    for _ in range(rounds):
        component_map, component_label = label_partition(label_map)
        areas = np.bincount(component_map.ravel())
        small = np.flatnonzero(areas < min_area)
        if not len(small) or len(areas) < 2:
            return label_map

        edges, counts = _adjacency(component_map)
        if not len(edges):
            return label_map

        order = np.lexsort((-counts, edges[:, 0]))
        edges = edges[order]
        first = np.concatenate(([True], edges[1:, 0] != edges[:-1, 0]))
        best_neighbour = np.full(len(areas), -1, dtype=np.int64)
        best_neighbour[edges[first, 0]] = edges[first, 1]

        new_label = component_label.copy()
        movable = small[best_neighbour[small] >= 0]
        if not len(movable):
            return label_map
        new_label[movable] = component_label[best_neighbour[movable]]
        label_map = new_label[component_map].astype(np.int32)

    return label_map


def _component_boxes(component_map, count):
    height, width = component_map.shape
    flat = component_map.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_ids = flat[order]
    starts = np.flatnonzero(np.concatenate(([True], sorted_ids[1:] != sorted_ids[:-1])))
    rows = order // width
    columns = order % width

    boxes = np.zeros((count, 4), dtype=np.int32)
    ids = sorted_ids[starts]
    boxes[ids, 0] = np.minimum.reduceat(rows, starts)
    boxes[ids, 1] = np.minimum.reduceat(columns, starts)
    boxes[ids, 2] = np.maximum.reduceat(rows, starts) + 1
    boxes[ids, 3] = np.maximum.reduceat(columns, starts) + 1
    return boxes


def _quantisation_error(rgb, labels, palette, sample=60000, seed=5):
    """Mean ΔE between pixels and the palette colour they were given."""
    flat = rgb.reshape(-1, 3)
    chosen = np.random.default_rng(seed).choice(len(flat), min(sample, len(flat)), replace=False)
    original = srgb_to_lab(flat[chosen].astype(np.float64) / 255.0)
    assigned = srgb_to_lab(palette[labels.ravel()[chosen]].astype(np.float64) / 255.0)
    return float(np.linalg.norm(original - assigned, axis=1).mean())


def _adaptive_palette(rgb, options):
    """Continuous-tone input: add colours until quantisation error is small.

    Flat artwork has a few exact colours; a gradient or a photograph has
    none, and any fixed count is either too coarse for one image or wasted
    on another. The count grows until the average error is under target.
    """
    # Choose the count on a small sample with short runs, then quantise once.
    flat = rgb.reshape(-1, 3)
    chosen = np.random.default_rng(9).choice(len(flat), min(20000, len(flat)), replace=False)
    sample = srgb_to_lab(flat[chosen].astype(np.float64) / 255.0)
    count = options.fallback_colors
    while count < options.max_fallback_colors:
        centres = _short_kmeans(sample, count)
        nearest = np.sqrt(squared_distances(sample, centres).min(axis=1))
        if float(nearest.mean()) <= options.target_delta:
            break
        count = min(options.max_fallback_colors, int(count * 1.6))
    label_map, palette = quantize_full_color(rgb, count)
    return label_map.astype(np.int32), palette


def _short_kmeans(points, count, iterations=8, seed=3):
    rng = np.random.default_rng(seed)
    centres = points[rng.choice(len(points), count, replace=False)]
    for _ in range(iterations):
        nearest = np.argmin(squared_distances(points, centres), axis=1)
        sizes = np.bincount(nearest, minlength=count)
        occupied = sizes > 0
        sums = np.stack([np.bincount(nearest, weights=points[:, c], minlength=count) for c in range(3)], axis=1)
        centres[occupied] = sums[occupied] / sizes[occupied][:, None]
    return centres


@dataclass
class Labelling:
    palette: np.ndarray
    label_map: np.ndarray
    mode: str
    noise: float  # capped at JPEG level, for tolerances
    raw_noise: float  # uncapped, for deciding whether to trace at all
    image: np.ndarray = None  # what the labels were computed from
    image_lab: np.ndarray = None  # the same in Lab, computed on first use
    reach: int = 1  # how far blends extend: 1 for antialiasing, 3 for JPEG


def build_partition(rgb, options, area_scale=1.0):
    return partition_labels(label_image(rgb, options), options, area_scale)


def label_image(rgb, options):
    """Palette and per-pixel labels: the expensive half, computed once."""
    # Noise widens tolerances up to JPEG levels and no further: a heavily
    # textured image isn't flat artwork with noise on it, and treating
    # 30 ΔE as "the same colour" would make it look like one.
    raw_noise = estimate_noise(rgb)
    noise = min(raw_noise, 0.6)
    pure_delta = options.pure_delta + 6.0 * noise
    rgb = denoise_texture(rgb, raw_noise)
    reach = getattr(options, "blend_reach", 1)
    palette, coverage = detect_palette(
        rgb,
        options.max_colors,
        options.min_palette_fraction,
        options.palette_merge_delta,
        pure_delta,
        noise,
        reach,
    )

    key = getattr(options, "key_colour", None)
    if key is not None and len(palette):
        # The key must be in the palette before labelling, so edge pixels
        # against transparency are read as blends with it.
        palette, _ = with_colour(palette, key)

    if len(palette) >= 2 and coverage >= options.min_flat_coverage:
        label_map, blends = assign_labels(rgb, palette, pure_delta, options.blend_residual, report=True, reach=reach)
        palette, label_map, _ = refine_palette(rgb, palette, label_map, blends, options, pure_delta, reach=reach)
        mode = "detected"
    else:
        label_map, palette = _adaptive_palette(rgb, options)
        mode = "kmeans"
    return Labelling(palette, label_map, mode, noise, raw_noise, rgb, reach=reach)


def with_colour(palette, colour, match_delta=3.0):
    """Palette containing `colour`, and its index. Reuses an entry within match_delta ΔE."""
    colour = np.asarray(colour, dtype=np.uint8).reshape(1, 3)
    if len(palette):
        distance = np.linalg.norm(
            srgb_to_lab(palette.astype(np.float64) / 255.0) - srgb_to_lab(colour / 255.0), axis=1
        )
        closest = int(np.argmin(distance))
        if float(distance[closest]) <= match_delta:
            return palette, closest
    return np.concatenate([palette, colour]).astype(np.uint8), len(palette)


def force_clear(labelling, alpha, key, solid=0.98):
    """Give the key's label to exactly the pixels under half opacity. Returns that label.

    A partly transparent pixel is labelled by the colour it was composited
    from, recovered by undoing the composite. Otherwise a soft edge reads
    as a band of colours halfway to the key, which no artwork contains.
    """
    palette, index = with_colour(labelling.palette, key)
    label_map = labelling.label_map.copy()
    clear = alpha < 0.5
    relabel = ~clear & ((label_map == index) | (alpha < solid))
    label_map[clear] = index
    others = np.delete(np.arange(len(palette)), index)
    if relabel.any() and len(others):
        opacity = alpha[relabel][:, None]
        composite = labelling.image[relabel].astype(np.float64)
        straight = (composite - (1.0 - opacity) * np.asarray(key, dtype=np.float64)) / opacity
        pixels = np.clip(np.round(straight), 0, 255).astype(np.uint8)[None]
        nearest, _ = _nearest_palette(pixels, palette[others])
        label_map[relabel] = others[nearest[0]]

    used = np.unique(np.append(label_map, index))
    remap = np.full(len(palette), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    labelling.palette = palette[used]
    labelling.label_map = remap[label_map]
    return int(remap[index])


def partition_labels(labelling, options, area_scale=1.0, start=None):
    """Merge undersized regions and index the result: the cheap half, repeatable.

    start: a label map already merged at a smaller scale, to carry on from
    rather than merge every region again from the beginning.
    """
    # In clean artwork a seven-pixel region is a star or a dot someone drew;
    # in a JPEG it is a speckle. The merge size follows the measured noise.
    blend = min(labelling.noise / 0.5, 1.0)
    min_area = round(options.min_region_area + blend * (options.noisy_region_area - options.min_region_area))
    palette, mode = labelling.palette, labelling.mode
    if labelling.image_lab is None:
        labelling.image_lab = lab_image(labelling.image)
    label_map = merge_small_regions(
        labelling.label_map if start is None else start,
        max(1, round(min_area * area_scale)),
        image_lab=labelling.image_lab,
        palette_lab=srgb_to_lab(palette.astype(np.float64) / 255.0),
        floor_area=max(1, round(options.min_region_area * area_scale)),
        keep_contrast=options.keep_contrast,
        image_rgb=labelling.image,
        palette_rgb=palette,
        sliver_reach=labelling.reach,
    )
    component_map, component_label = label_partition(label_map)
    count = len(component_label)
    areas = np.bincount(component_map.ravel(), minlength=count)

    return Partition(
        palette=palette,
        label_map=label_map,
        component_map=component_map,
        component_label=component_label.astype(np.int32),
        areas=areas,
        boxes=_component_boxes(component_map, count),
        palette_mode=mode,
    )
