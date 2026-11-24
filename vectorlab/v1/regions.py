"""Stage 2+3: region segmentation and contour extraction.

Connected components use run-based labelling with union-find, which is
the same shape as the classic two-pass CCL and ports directly to C++.

Contours are extracted on the *pixel grid*, not through pixel centres:
every boundary is a unit-length edge between an inside and an outside
pixel. That yields exact staircase rings with integer coordinates -
the same starting point Potrace uses before it simplifies.
"""

import numpy as np


class _UnionFind:
    def __init__(self):
        self._parent = []

    def make(self):
        self._parent.append(len(self._parent))
        return len(self._parent) - 1

    def find(self, node):
        root = node
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]
        return root

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[max(left_root, right_root)] = min(left_root, right_root)


def _row_runs(row_mask):
    """Horizontal runs of True as (start, end_exclusive) pairs."""
    padded = np.concatenate(([False], row_mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2]))


def label_components(mask):
    """4-connected labelling. Returns (label_map int32, component_count).

    Background is -1; components are numbered from 0.
    """
    height, width = mask.shape
    labels = np.full((height, width), -1, dtype=np.int32)
    union_find = _UnionFind()
    previous_runs = []

    for y in range(height):
        current_runs = []
        for start, end in _row_runs(mask[y]):
            run_label = -1
            for previous_start, previous_end, previous_label in previous_runs:
                if previous_start < end and start < previous_end:
                    if run_label == -1:
                        run_label = previous_label
                    else:
                        union_find.union(run_label, previous_label)
            if run_label == -1:
                run_label = union_find.make()
            labels[y, start:end] = run_label
            current_runs.append((start, end, run_label))
        previous_runs = current_runs

    if not union_find._parent:
        return labels, 0

    roots = np.array([union_find.find(i) for i in range(len(union_find._parent))])
    unique_roots, compact = np.unique(roots, return_inverse=True)
    remap = np.concatenate(([-1], compact)).astype(np.int32)
    return remap[labels + 1], len(unique_roots)


def extract_rings(mask):
    """Closed integer rings bounding the True pixels of `mask`.

    Returns a list of rings; each ring is an (n, 2) float array of
    pixel-grid corner coordinates. Outer rings and holes are both
    present - orientation (via signed area) tells them apart, and an
    even-odd fill renders them correctly without extra work.
    """
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask

    solid = padded[1:-1, 1:-1]
    up_open = solid & ~padded[:-2, 1:-1]
    down_open = solid & ~padded[2:, 1:-1]
    left_open = solid & ~padded[1:-1, :-2]
    right_open = solid & ~padded[1:-1, 2:]

    # Directed unit edges, wound so the solid side stays consistent.
    outgoing = {}

    def add_edge(from_point, to_point):
        outgoing.setdefault(from_point, []).append(to_point)

    for y, x in zip(*np.nonzero(up_open)):
        add_edge((x, y), (x + 1, y))
    for y, x in zip(*np.nonzero(right_open)):
        add_edge((x + 1, y), (x + 1, y + 1))
    for y, x in zip(*np.nonzero(down_open)):
        add_edge((x + 1, y + 1), (x, y + 1))
    for y, x in zip(*np.nonzero(left_open)):
        add_edge((x, y + 1), (x, y))

    rings = []
    for start_point in list(outgoing.keys()):
        while outgoing.get(start_point):
            ring = [start_point]
            current = outgoing[start_point].pop()
            while current != start_point:
                ring.append(current)
                candidates = outgoing.get(current)
                if not candidates:
                    ring = []
                    break
                current = candidates.pop()
            if len(ring) >= 4:
                rings.append(np.array(ring, dtype=np.float64))

    return rings


def signed_area(ring):
    x = ring[:, 0]
    y = ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
