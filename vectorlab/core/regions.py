"""Connected regions of a label map, and per-colour helpers shared by the tools."""

import numpy as np

from .colour import srgb_to_lab


class _UnionFind:
    def __init__(self):
        self.parent = []

    def make(self):
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def find(self, node):
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            self.parent[node], node = root, self.parent[node]
        return root

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _row_runs(row):
    """(start, end_exclusive, value) for each run of equal values."""
    change = np.flatnonzero(row[1:] != row[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(row)]))
    return list(zip(starts.tolist(), ends.tolist(), row[starts].tolist()))


def label_partition(value_map, connectivity=4):
    """Connected components of equal value. Returns (component_map, component_value).

    Components are numbered in the order they're first met scanning rows,
    the same as `_label_partition_sweep`, which this replaces for speed: runs
    of equal value along each row are joined to the runs they touch in the
    next row, all at once, then merged by repeatedly pointing each run at
    the smallest run it connects to.
    """
    if value_map.size == 0:
        return np.zeros(value_map.shape, dtype=np.int32), np.zeros(0, dtype=value_map.dtype)
    starts = np.ones(value_map.shape, dtype=bool)
    starts[:, 1:] = value_map[:, 1:] != value_map[:, :-1]
    run = (np.cumsum(starts.ravel()) - 1).reshape(value_map.shape)

    pairs = [(run[:-1][value_map[:-1] == value_map[1:]], run[1:][value_map[:-1] == value_map[1:]])]
    if connectivity == 8:
        down_right = value_map[:-1, :-1] == value_map[1:, 1:]
        down_left = value_map[:-1, 1:] == value_map[1:, :-1]
        pairs.append((run[:-1, :-1][down_right], run[1:, 1:][down_right]))
        pairs.append((run[:-1, 1:][down_left], run[1:, :-1][down_left]))
    first = np.concatenate([a for a, _ in pairs])
    second = np.concatenate([b for _, b in pairs])

    root = np.arange(int(run[-1, -1]) + 1)
    while True:
        low, high = np.minimum(root[first], root[second]), np.maximum(root[first], root[second])
        joined = low != high
        if not joined.any():
            break
        np.minimum.at(root, high[joined], low[joined])
        while True:
            jumped = root[root]
            if np.array_equal(jumped, root):
                break
            root = jumped

    _, compact = np.unique(root, return_inverse=True)
    component_map = compact.ravel().astype(np.int32)[run]
    count = int(component_map.max()) + 1
    component_value = np.zeros(count, dtype=value_map.dtype)
    component_value[component_map.ravel()] = value_map.ravel()
    return component_map, component_value


def _label_partition_sweep(value_map, connectivity=4):
    """Reference implementation: a two-pointer sweep over each row's runs."""
    height = value_map.shape[0]
    components = np.empty(value_map.shape, dtype=np.int32)
    union_find = _UnionFind()
    reach = 1 if connectivity == 8 else 0
    previous = []

    for y in range(height):
        current = []
        pointer = 0
        for start, end, value in _row_runs(value_map[y]):
            run_id = -1
            while pointer < len(previous) and previous[pointer][1] + reach <= start:
                pointer += 1
            probe = pointer
            while probe < len(previous) and previous[probe][0] < end + reach:
                previous_start, previous_end, previous_value, previous_id = previous[probe]
                if previous_value == value:
                    if run_id == -1:
                        run_id = previous_id
                    else:
                        union_find.union(run_id, previous_id)
                probe += 1
            if run_id == -1:
                run_id = union_find.make()
            components[y, start:end] = run_id
            current.append((start, end, value, run_id))
        previous = current

    roots = np.array([union_find.find(node) for node in range(len(union_find.parent))])
    _, compact = np.unique(roots, return_inverse=True)
    component_map = compact.astype(np.int32)[components]

    count = int(component_map.max()) + 1 if component_map.size else 0
    component_value = np.zeros(count, dtype=value_map.dtype)
    component_value[component_map.ravel()] = value_map.ravel()
    return component_map, component_value


def unique_colours(rgb):
    """(distinct colours (n, 3) uint8, index of each pixel's colour). Artwork has far fewer
    colours than pixels, so per-colour work done once per colour is much cheaper."""
    flat = rgb.reshape(-1, 3).astype(np.int32)
    keys = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    colours = np.stack([(unique_keys >> 16) & 255, (unique_keys >> 8) & 255, unique_keys & 255], axis=1)
    return colours.astype(np.uint8), inverse.ravel()


def lab_image(rgb):
    """The image in Lab, converting each distinct colour once."""
    colours, inverse = unique_colours(rgb)
    return srgb_to_lab(colours.astype(np.float64) / 255.0)[inverse].reshape(rgb.shape[:-1] + (3,))
