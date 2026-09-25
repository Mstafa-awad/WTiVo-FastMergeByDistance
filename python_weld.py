from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


def python_weld(vertices, faces, distance, centroid_merge=False, remove_degenerate=True):
    """Portable CPU fallback using a spatial hash and NumPy arrays."""
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.uint32)
    if distance <= 0:
        reps = np.arange(len(vertices), dtype=np.uint32)
        return vertices.copy(), faces.copy(), reps

    d2 = float(distance) * float(distance)
    inv = 1.0 / float(distance)
    buckets = defaultdict(list)
    parent = np.arange(len(vertices), dtype=np.int64)
    rank = np.zeros(len(vertices), dtype=np.uint8)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for i, p in enumerate(vertices):
        cx = math.floor(float(p[0]) * inv)
        cy = math.floor(float(p[1]) * inv)
        cz = math.floor(float(p[2]) * inv)
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for j in buckets.get((cx + dx, cy + dy, cz + dz), ()):
                        delta = p - vertices[j]
                        if float(np.dot(delta, delta)) <= d2:
                            union(i, j)
        buckets[(cx, cy, cz)].append(i)

    roots = np.empty(len(vertices), dtype=np.int64)
    for i in range(len(vertices)):
        roots[i] = find(i)

    unique_roots, component = np.unique(roots, return_inverse=True)
    count = len(unique_roots)
    reps = np.empty(count, dtype=np.uint32)

    if centroid_merge:
        sums = np.zeros((count, 3), dtype=np.float64)
        counts = np.bincount(component, minlength=count).astype(np.int64)
        np.add.at(sums, component, vertices.astype(np.float64))
        out_vertices = (sums / counts[:, None]).astype(np.float32)
        best = np.full(count, np.inf, dtype=np.float64)
        for i, cid in enumerate(component):
            d = float(np.sum((vertices[i] - out_vertices[cid]) ** 2))
            if d < best[cid]:
                best[cid] = d
                reps[cid] = np.uint32(i)
    else:
        first = np.full(count, len(vertices), dtype=np.int64)
        np.minimum.at(first, component, np.arange(len(vertices), dtype=np.int64))
        reps = first.astype(np.uint32)
        out_vertices = np.ascontiguousarray(vertices[reps])

    mapped = component[faces]
    if remove_degenerate:
        valid = (
            (mapped[:, 0] != mapped[:, 1]) &
            (mapped[:, 1] != mapped[:, 2]) &
            (mapped[:, 2] != mapped[:, 0])
        )
        mapped = mapped[valid]
    return (
        np.ascontiguousarray(out_vertices, dtype=np.float32),
        np.ascontiguousarray(mapped, dtype=np.uint32),
        np.ascontiguousarray(reps, dtype=np.uint32),
        np.ascontiguousarray(component, dtype=np.uint32),
    )
