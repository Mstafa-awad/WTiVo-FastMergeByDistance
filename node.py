from __future__ import annotations

import io
import logging
import math
import struct
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from comfy_api.latest import Types

from .native import native_available, native_thread_count, native_weld
from .python_weld import python_weld

LOGGER = logging.getLogger("WTiVo-FastMergeByDistance")
VERSION = "8.0.0"

# glTF constants.
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
TRIANGLES = 4

_COMPONENT_INFO = {
    5120: (1, "b"),   # BYTE
    5121: (1, "B"),   # UNSIGNED BYTE
    5122: (2, "h"),   # SHORT
    5123: (2, "H"),   # UNSIGNED SHORT
    5125: (4, "I"),   # UNSIGNED INT
    5126: (4, "f"),   # FLOAT
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


@dataclass
class GLB:
    header: bytes
    chunks: list[tuple[int, bytes]]
    json_obj: dict[str, Any]
    bin_index: int | None


class GLBError(RuntimeError):
    pass


def _align4(n: int) -> int:
    return (n + 3) & ~3


def _parse_glb(data: bytes) -> GLB:
    if len(data) < 12:
        raise GLBError("File is too small to be a GLB.")
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise GLBError("Only glTF 2.0 GLB files are supported.")
    if length > len(data):
        raise GLBError("GLB declares a length larger than the supplied data.")

    chunks: list[tuple[int, bytes]] = []
    offset = 12
    while offset + 8 <= length:
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        end = offset + chunk_len
        if end > length:
            raise GLBError("GLB chunk extends beyond the file length.")
        chunks.append((chunk_type, bytes(data[offset:end])))
        offset = end

    json_chunk = next((payload for typ, payload in chunks if typ == CHUNK_JSON), None)
    if json_chunk is None:
        raise GLBError("GLB has no JSON chunk.")
    try:
        import json
        json_obj = json.loads(json_chunk.decode("utf-8").rstrip(" \t\r\n\x00"))
    except Exception as exc:
        raise GLBError(f"Invalid GLB JSON: {exc}") from exc

    bin_positions = [i for i, (typ, _) in enumerate(chunks) if typ == CHUNK_BIN]
    if len(bin_positions) > 1:
        raise GLBError("GLB contains multiple BIN chunks; this node expects one.")
    return GLB(header=data[:12], chunks=chunks, json_obj=json_obj,
               bin_index=bin_positions[0] if bin_positions else None)


def _rebuild_glb(glb: GLB) -> bytes:
    import json

    new_chunks = list(glb.chunks)
    json_bytes = json.dumps(glb.json_obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((_align4(len(json_bytes))) - len(json_bytes))

    json_index = next((i for i, (t, _) in enumerate(new_chunks) if t == CHUNK_JSON), None)
    if json_index is None:
        raise GLBError("Cannot rebuild GLB without a JSON chunk.")
    new_chunks[json_index] = (CHUNK_JSON, json_bytes)

    total = 12 + sum(8 + len(payload) for _, payload in new_chunks)
    out = bytearray(struct.pack("<4sII", b"glTF", 2, total))
    for typ, payload in new_chunks:
        out += struct.pack("<II", len(payload), typ)
        out += payload
    return bytes(out)


def _get_bin(glb: GLB) -> bytes:
    if glb.bin_index is None:
        raise GLBError("GLB has no BIN chunk.")
    return glb.chunks[glb.bin_index][1]


def _buffer_slice(glb: GLB, accessor_index: int, want_indices: bool = False) -> tuple[bytes, dict[str, Any], dict[str, Any], int]:
    accessors = glb.json_obj.get("accessors", [])
    views = glb.json_obj.get("bufferViews", [])
    buffers = glb.json_obj.get("buffers", [])
    if accessor_index < 0 or accessor_index >= len(accessors):
        raise GLBError(f"Accessor {accessor_index} is out of range.")
    accessor = accessors[accessor_index]
    if "sparse" in accessor:
        raise GLBError("Sparse accessors are not supported by the fast GLB path.")
    view_index = accessor.get("bufferView")
    if view_index is None:
        raise GLBError("Accessor without bufferView is not supported for mesh welding.")
    view = views[view_index]
    buffer_index = view.get("buffer", 0)
    if buffer_index != 0 or ("uri" in buffers[buffer_index]):
        raise GLBError("Mesh data must be in the embedded GLB BIN buffer.")
    bin_data = _get_bin(glb)
    comp_size, _ = _COMPONENT_INFO[accessor["componentType"]]
    components = _TYPE_COMPONENTS[accessor["type"]]
    element_size = comp_size * components
    view_offset = int(view.get("byteOffset", 0))
    acc_offset = int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", element_size))
    count = int(accessor["count"])
    start = view_offset + acc_offset
    end = start + (stride * (count - 1) if count else 0) + element_size
    if end > len(bin_data):
        raise GLBError("Accessor points outside the BIN chunk.")
    return bin_data[start:end], accessor, view, stride


def _decode_accessor(glb: GLB, accessor_index: int) -> np.ndarray:
    raw, acc, _view, stride = _buffer_slice(glb, accessor_index)
    comp_size, fmt = _COMPONENT_INFO[acc["componentType"]]
    comps = _TYPE_COMPONENTS[acc["type"]]
    count = int(acc["count"])
    # numpy dtype mapping explicitly avoids platform-endian surprises.
    dtype_map = {
        5120: np.dtype("<i1"), 5121: np.dtype("<u1"), 5122: np.dtype("<i2"),
        5123: np.dtype("<u2"), 5125: np.dtype("<u4"), 5126: np.dtype("<f4"),
    }
    dtype = dtype_map[acc["componentType"]]
    out = np.empty((count, comps), dtype=dtype)
    for i in range(count):
        begin = i * stride
        out[i] = np.frombuffer(raw, dtype=dtype, count=comps, offset=begin)
    if comps == 1:
        return out[:, 0]
    return out


def _decode_indices(glb: GLB, accessor_index: int) -> np.ndarray:
    arr = _decode_accessor(glb, accessor_index)
    if arr.ndim != 1:
        raise GLBError("Index accessor must be SCALAR.")
    return np.ascontiguousarray(arr, dtype=np.uint32)


def _append_binary(glb: GLB, payload: bytes) -> tuple[int, int]:
    if glb.bin_index is None:
        glb.chunks.append((CHUNK_BIN, b""))
        glb.bin_index = len(glb.chunks) - 1
    old = glb.chunks[glb.bin_index][1]
    pad = b"\x00" * ((_align4(len(old))) - len(old))
    start = len(old) + len(pad)
    payload_padded = payload + (b"\x00" * ((_align4(len(payload))) - len(payload)))
    new_bin = old + pad + payload_padded
    glb.chunks[glb.bin_index] = (CHUNK_BIN, new_bin)
    buffers = glb.json_obj.setdefault("buffers", [])
    if buffers:
        buffers[0]["byteLength"] = len(new_bin)
    return start, len(payload)


def _make_accessor(glb: GLB, template_acc: dict[str, Any], template_view: dict[str, Any], raw: bytes, count: int,
                   *, is_index: bool = False, min_max: tuple[list[float], list[float]] | None = None) -> int:
    views = glb.json_obj.setdefault("bufferViews", [])
    accessors = glb.json_obj.setdefault("accessors", [])

    start, byte_len = _append_binary(glb, raw)
    new_view = {
        "buffer": 0,
        "byteOffset": start,
        "byteLength": byte_len,
        "target": 34963 if is_index else 34962,
    }
    if "byteStride" in template_view and not is_index:
        # New data is tightly packed, so byteStride is deliberately omitted.
        pass
    views.append(new_view)
    view_index = len(views) - 1

    new_acc = {k: v for k, v in template_acc.items() if k not in ("bufferView", "byteOffset", "count", "min", "max", "sparse")}
    new_acc["bufferView"] = view_index
    new_acc["byteOffset"] = 0
    new_acc["count"] = int(count)
    if min_max is not None:
        new_acc["min"] = min_max[0]
        new_acc["max"] = min_max[1]
    accessors.append(new_acc)
    return len(accessors) - 1


def _encode_selected(original: np.ndarray, representatives: np.ndarray) -> bytes:
    selected = np.ascontiguousarray(original[representatives])
    return selected.tobytes(order="C")


def _encode_indices(indices: np.ndarray, component_type: int) -> bytes:
    limits = {5121: 255, 5123: 65535, 5125: 0xFFFFFFFF}
    if component_type not in limits:
        component_type = 5125
    max_index = int(indices.max()) if indices.size else 0
    if component_type != 5125 and max_index > limits[component_type]:
        component_type = 5125
    dtype_map = {5121: np.dtype("<u1"), 5123: np.dtype("<u2"), 5125: np.dtype("<u4")}
    return np.ascontiguousarray(indices, dtype=dtype_map[component_type]).tobytes(order="C"), component_type


def _update_minmax_position(acc: dict[str, Any], vertices: np.ndarray) -> tuple[list[float], list[float]]:
    if vertices.size == 0:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    mn = vertices.min(axis=0).astype(np.float64).tolist()
    mx = vertices.max(axis=0).astype(np.float64).tolist()
    return [float(x) for x in mn], [float(x) for x in mx]


def _primitive_is_supported(glb: GLB, primitive: dict[str, Any]) -> bool:
    if int(primitive.get("mode", TRIANGLES)) != TRIANGLES:
        return False
    if "indices" not in primitive:
        return False
    if "POSITION" not in primitive.get("attributes", {}):
        return False
    ext = primitive.get("extensions", {})
    unsupported = {"KHR_draco_mesh_compression", "EXT_meshopt_compression"}
    return not any(e in ext for e in unsupported)


def _primitive_data(glb: GLB, primitive: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pos_idx = int(primitive["attributes"]["POSITION"])
    idx_idx = int(primitive["indices"])
    pos = _decode_accessor(glb, pos_idx)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise GLBError("POSITION accessor is not VEC3.")
    pos = np.ascontiguousarray(pos, dtype=np.float32)
    idx = _decode_indices(glb, idx_idx).reshape(-1, 3)
    return pos, idx


def _iter_primitives(glb: GLB):
    for mesh_index, mesh in enumerate(glb.json_obj.get("meshes", [])):
        for prim_index, primitive in enumerate(mesh.get("primitives", [])):
            yield mesh_index, prim_index, mesh, primitive


def _edge_stats(faces: np.ndarray) -> tuple[int, int, int]:
    if faces.size == 0:
        return 0, 0, 0
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    # Lexicographically sort edges and count runs.
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    same = np.all(sorted_edges[1:] == sorted_edges[:-1], axis=1)
    starts = np.concatenate(([True], ~same))
    ends = np.concatenate((~same, [True]))
    idx = np.flatnonzero(starts)
    end_idx = np.flatnonzero(ends)
    counts = end_idx - idx + 1
    # Safety fallback for unusual zero-length edge sets.
    bad_boundary = int(np.count_nonzero(counts == 1))
    bad_over = int(np.count_nonzero(counts > 2))
    return bad_boundary, bad_over, int(bad_boundary + bad_over)


def _vertex_fan_bad(faces: np.ndarray, vertex_count: int) -> int:
    """Return a count of vertices with disconnected local fans.

    This catches bow-tie/non-manifold vertices that can survive an edge-count
    check. The implementation is intentionally used as a second-stage check;
    for huge meshes the edge count is the dominant inexpensive test.
    """
    if faces.size == 0 or vertex_count == 0:
        return 0
    incident: list[list[tuple[int, int]]] = [[] for _ in range(vertex_count)]
    for a, b, c in faces.tolist():
        incident[a].append((b, c))
        incident[b].append((c, a))
        incident[c].append((a, b))

    bad = 0
    for v, links in enumerate(incident):
        if len(links) < 2:
            continue
        adjacency: dict[int, set[int]] = {}
        for left, right in links:
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)
        start = next(iter(adjacency), None)
        if start is None:
            continue
        seen = {start}
        stack = [start]
        while stack:
            n = stack.pop()
            for q in adjacency.get(n, ()):
                if q not in seen:
                    seen.add(q)
                    stack.append(q)
        if len(seen) != len(adjacency):
            bad += 1
    return bad


def _validate_watertight(faces: np.ndarray, vertex_count: int) -> tuple[bool, dict[str, int]]:
    boundary, over, bad_edges = _edge_stats(faces)
    # Fan validation is most valuable once every edge has exactly two uses.
    bad_fans = 0 if bad_edges else _vertex_fan_bad(faces, vertex_count)
    ok = bad_edges == 0 and bad_fans == 0 and len(faces) > 0
    return ok, {
        "boundary_edges": boundary,
        "overfull_edges": over,
        "bad_edge_groups": bad_edges,
        "bad_vertex_fans": bad_fans,
    }



def _load_primitive_attributes(glb: GLB, primitive: dict[str, Any], vertex_count: int):
    """Decode every per-vertex attribute so a positional weld never overwrites seams."""
    accessors = glb.json_obj.get("accessors", [])
    attrs = primitive.get("attributes", {})
    arrays: list[tuple[str, np.ndarray]] = []

    for semantic, accessor_index in attrs.items():
        if semantic == "POSITION":
            continue
        arr = _decode_accessor(glb, int(accessor_index))
        if int(accessors[int(accessor_index)].get("count", 0)) != vertex_count:
            raise GLBError(f"Attribute {semantic} count mismatch.")
        arrays.append((semantic, np.ascontiguousarray(arr)))

    for ti, target in enumerate(primitive.get("targets", [])):
        for semantic, accessor_index in target.items():
            arr = _decode_accessor(glb, int(accessor_index))
            if int(accessors[int(accessor_index)].get("count", 0)) != vertex_count:
                raise GLBError(f"Morph target {ti} {semantic} count mismatch.")
            arrays.append((f"__MORPH_{ti}_{semantic}", np.ascontiguousarray(arr)))
    return arrays


def _has_attribute_seams_or_texture(glb: GLB, primitive: dict[str, Any]) -> bool:
    attrs = primitive.get("attributes", {})
    # UVs/hard normals/tangents/colors/skin data all make glTF render vertices
    # semantically distinct even when their XYZ positions coincide.
    if any(k != "POSITION" for k in attrs):
        return True
    if primitive.get("targets"):
        return True
    mat_index = primitive.get("material")
    if mat_index is not None:
        mats = glb.json_obj.get("materials", [])
        if 0 <= int(mat_index) < len(mats):
            m = mats[int(mat_index)]
            pbr = m.get("pbrMetallicRoughness", {})
            if pbr.get("baseColorTexture") or pbr.get("metallicRoughnessTexture"):
                return True
            if m.get("normalTexture") or m.get("occlusionTexture") or m.get("emissiveTexture"):
                return True
    return False


def _build_attribute_safe_vertices(
    glb: GLB,
    primitive: dict[str, Any],
    original_vertices: np.ndarray,
    original_faces: np.ndarray,
    cluster_map: np.ndarray,
    welded_vertices: np.ndarray,
    remove_degenerate: bool = True,
):
    """Create glTF render vertices keyed by welded position cluster + attributes.

    All members of a positional cluster receive the same XYZ, but vertices with
    different UVs/normals/tangents/colors/joints/weights/morph values remain
    separate. This prevents texture and shading corruption at seams.
    """
    n = len(original_vertices)
    attr_arrays = _load_primitive_attributes(glb, primitive, n)

    fields = [("cluster", np.dtype("<u4"))]
    for name, arr in attr_arrays:
        # One fixed-size field per accessor. Structured dtypes let numpy compare
        # whole attribute records efficiently without Python tuple allocation.
        flat_shape = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
        fields.append((name, arr.dtype, (flat_shape,)))

    keys = np.empty(n, dtype=np.dtype(fields))
    keys["cluster"] = np.asarray(cluster_map, dtype=np.uint32)
    for name, arr in attr_arrays:
        keys[name] = arr.reshape(n, -1)

    # unique_keys are canonical render-vertex records. Return_inverse maps each
    # original vertex to its attribute-safe render vertex.
    _unique_keys, render_for_original = np.unique(keys, return_inverse=True)
    render_for_original = np.asarray(render_for_original, dtype=np.uint32)

    # Cluster position, not the original position, is authoritative after weld.
    out_vertices = np.asarray(welded_vertices, dtype=np.float32)[
        render_for_original if False else _unique_keys["cluster"]
    ].copy()

    new_faces = render_for_original[original_faces]

    # Degeneracy must be checked on the geometric cluster, not the render index,
    # because two UV-seam vertices may intentionally have different indices at
    # exactly the same XYZ position.
    geometric_faces = cluster_map[original_faces]
    if remove_degenerate:
        valid = (
            (geometric_faces[:, 0] != geometric_faces[:, 1])
            & (geometric_faces[:, 1] != geometric_faces[:, 2])
            & (geometric_faces[:, 2] != geometric_faces[:, 0])
        )
        new_faces = new_faces[valid]
    new_faces = np.ascontiguousarray(new_faces, dtype=np.uint32)
    return (
        np.ascontiguousarray(out_vertices, dtype=np.float32),
        new_faces,
        render_for_original,
        attr_arrays,
    )


def _rewrite_primitive_attributes(
    glb: GLB,
    primitive: dict[str, Any],
    render_for_original: np.ndarray,
    new_indices: np.ndarray,
    new_vertex_positions: np.ndarray,
    attr_arrays: list[tuple[str, np.ndarray]],
    original_vertex_count: int,
) -> None:
    accessors = glb.json_obj.get("accessors", [])
    views = glb.json_obj.get("bufferViews", [])
    attrs = primitive.get("attributes", {})

    # render_for_original maps original vertex -> final glTF render vertex.
    # Build inverse representatives by selecting the first original vertex for
    # every final render vertex. Attribute-safe keys guarantee that every source
    # in a render vertex has identical per-vertex attributes.
    render_count = len(new_vertex_positions)
    reps = np.full(render_count, -1, dtype=np.int64)
    for src, dst in enumerate(render_for_original.tolist()):
        if reps[dst] < 0:
            reps[dst] = src
    if np.any(reps < 0):
        raise GLBError("Internal attribute-safe remap produced an unused render vertex.")
    reps = reps.astype(np.uint32)

    # Decode/write the original primitive attributes using the final render
    # representatives. POSITION gets the welded XYZ values directly.
    attr_lookup = {name: arr for name, arr in attr_arrays}
    for semantic, accessor_index in list(attrs.items()):
        ai = int(accessor_index)
        original = _decode_accessor(glb, ai)
        _raw, template_acc, template_view, _stride = _buffer_slice(glb, ai)
        if int(template_acc.get("count", 0)) != original_vertex_count:
            raise GLBError(f"Attribute {semantic} count mismatch.")
        if semantic == "POSITION":
            selected = np.ascontiguousarray(new_vertex_positions, dtype=np.float32)
            min_max = _update_minmax_position(template_acc, selected)
        else:
            selected = np.ascontiguousarray(original[reps])
            min_max = None
        new_ai = _make_accessor(
            glb,
            template_acc,
            template_view,
            selected.tobytes(order="C"),
            render_count,
            is_index=False,
            min_max=min_max,
        )
        primitive["attributes"][semantic] = new_ai

    # Morph target attributes are keyed into attr_lookup only for identification;
    # write them by recovering their original accessors through the target list.
    for ti, target in enumerate(primitive.get("targets", [])):
        for semantic, accessor_index in list(target.items()):
            ai = int(accessor_index)
            original = _decode_accessor(glb, ai)
            _raw, template_acc, template_view, _stride = _buffer_slice(glb, ai)
            if int(template_acc.get("count", 0)) != original_vertex_count:
                raise GLBError(f"Morph target {ti} {semantic} count mismatch.")
            selected = np.ascontiguousarray(original[reps])
            new_ai = _make_accessor(
                glb,
                template_acc,
                template_view,
                selected.tobytes(order="C"),
                render_count,
                is_index=False,
                min_max=None,
            )
            target[semantic] = new_ai

    old_indices_index = int(primitive["indices"])
    old_indices_acc = accessors[old_indices_index]
    old_indices_view = views[old_indices_acc["bufferView"]]
    index_bytes, component_type = _encode_indices(
        new_indices.reshape(-1), int(old_indices_acc["componentType"])
    )
    new_index_acc = dict(old_indices_acc)
    new_index_acc["componentType"] = component_type
    new_index_acc.pop("min", None)
    new_index_acc.pop("max", None)
    new_ii = _make_accessor(
        glb,
        new_index_acc,
        old_indices_view,
        index_bytes,
        int(new_indices.size),
        is_index=True,
    )
    primitive["indices"] = new_ii


def _representatives_from_cluster(cluster_map: np.ndarray, original_vertex_count: int, output_vertex_count: int) -> np.ndarray:
    reps = np.full(output_vertex_count, -1, dtype=np.int64)
    for src, dst in enumerate(np.asarray(cluster_map, dtype=np.int64).tolist()):
        if reps[dst] < 0:
            reps[dst] = src
    if np.any(reps < 0):
        raise GLBError("Internal weld cluster produced an unused output vertex.")
    return reps.astype(np.uint32)


def _weld_primitive(
    glb: GLB,
    primitive: dict[str, Any],
    vertices: np.ndarray,
    faces: np.ndarray,
    distance: float,
    centroid_merge: bool,
    remove_degenerate: bool,
    force_strict: bool = False,
):
    result = native_weld(
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.uint32),
        distance,
        centroid_merge,
        remove_degenerate,
    )
    if result is None:
        new_vertices, new_faces, representatives, cluster_map = python_weld(
            np.ascontiguousarray(vertices, dtype=np.float32),
            np.ascontiguousarray(faces, dtype=np.uint32),
            distance,
            centroid_merge,
            remove_degenerate,
        )
        used_native = False
    else:
        new_vertices, new_faces, representatives, cluster_map = result
        used_native = True

    if _has_attribute_seams_or_texture(glb, primitive) and not force_strict:
        safe_vertices, safe_faces, render_for_original, attr_arrays = _build_attribute_safe_vertices(
            glb, primitive, vertices, faces, cluster_map, new_vertices, remove_degenerate
        )
        return safe_vertices, safe_faces, cluster_map, render_for_original, attr_arrays, used_native

    # Plain geometry: one output vertex per weld cluster, preserving the fast
    # strict-topology behavior.
    return (
        np.ascontiguousarray(new_vertices, dtype=np.float32),
        np.ascontiguousarray(new_faces, dtype=np.uint32),
        np.ascontiguousarray(cluster_map, dtype=np.uint32),
        None,
        None,
        used_native,
    )


def _edge_stats_with_map(faces: np.ndarray, edge_vertex_map: np.ndarray) -> tuple[int, int, int]:
    if faces.size == 0:
        return 0, 0, 0
    mapped = edge_vertex_map[faces]
    return _edge_stats(mapped)


def _validate_geometric_watertight(
    faces: np.ndarray,
    vertex_count: int,
    geometric_map: np.ndarray,
) -> tuple[bool, dict[str, int]]:
    mapped = np.asarray(geometric_map[faces], dtype=np.uint32)
    if mapped.size:
        valid = (
            (mapped[:, 0] != mapped[:, 1])
            & (mapped[:, 1] != mapped[:, 2])
            & (mapped[:, 2] != mapped[:, 0])
        )
        mapped = mapped[valid]
    boundary, over, bad_edges = _edge_stats(mapped)
    bad_fans = 0 if bad_edges else _vertex_fan_bad(mapped, vertex_count)
    ok = bad_edges == 0 and bad_fans == 0 and len(mapped) > 0
    return ok, {
        "boundary_edges": boundary,
        "overfull_edges": over,
        "bad_edge_groups": bad_edges,
        "bad_vertex_fans": bad_fans,
    }

def _primitive_is_watertight(glb: GLB, primitive: dict[str, Any]) -> tuple[bool, dict[str, int], np.ndarray, np.ndarray]:
    vertices, faces = _primitive_data(glb, primitive)
    ok, stats = _validate_watertight(faces, len(vertices))
    return ok, stats, vertices, faces


def _adaptive_distance_step(base_step: float, face_count: int) -> float:
    """Choose a safer retry increment for mesh size.

    Large meshes are more sensitive to over-welding, so use a smaller increment.
    Smaller meshes keep the normal user-entered increment.
    """
    if face_count >= 1_000_000:
        return base_step * 0.25
    if face_count >= 250_000:
        return base_step * 0.5
    return base_step


class WTiVoFastMergeByDistance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FILE_3D_GLB",),
                "merge_distance": (
                    "FLOAT",
                    {
                        "default": 0.0001,
                        "min": 0.00000001,
                        "max": 10.0,
                        "step": 0.00001,
                    },
                ),
                "centroid_merge": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Center of cluster",
                        "label_off": "Closest original vertex",
                    },
                ),
                "remove_degenerate": ("BOOLEAN", {"default": True}),
                "topology_mode": (["TEXTURE_SAFE", "STRICT_BLENDER"], {"default": "TEXTURE_SAFE"}),
                "adaptive_retry_step": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Auto step by polygon count",
                    "label_off": "Use exact entered step",
                }),
            },
            "optional": {
                "max_attempts": (
                    "INT",
                    {"default": 100, "min": 1, "max": 10000, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("FILE_3D_GLB",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "3D/WTiVo"
    DESCRIPTION = (
        "CPU geometric watertight welding for GLB meshes. Separates geometric "
        "closure from glTF render-vertex seams, skips meshes that are already "
        "geometrically closed, and retries distance as step, 2x step, 3x step... "
        "without treating UV seam duplicates as broken geometry. Failed retries never "
        "crash the workflow; the exact original GLB is returned instead."
    )

    def execute(
        self,
        model: Types.File3D,
        merge_distance: float,
        centroid_merge: bool = False,
        remove_degenerate: bool = True,
        topology_mode: str = "TEXTURE_SAFE",
        adaptive_retry_step: bool = True,
        max_attempts: int = 100,
    ):
        start = time.perf_counter()
        distance_step = float(merge_distance)
        if not math.isfinite(distance_step) or distance_step <= 0:
            raise ValueError("merge_distance must be a finite value greater than 0.")
        max_attempts = max(1, int(max_attempts))
        topology_mode = str(topology_mode).upper()
        if topology_mode not in {"TEXTURE_SAFE", "STRICT_BLENDER"}:
            raise ValueError("topology_mode must be TEXTURE_SAFE or STRICT_BLENDER")

        if native_available():
            LOGGER.info("[WTiVo Fast Merge] Native C++ enabled | OpenMP threads=%d", native_thread_count())
        else:
            LOGGER.warning("[WTiVo Fast Merge] Native C++ DLL not found; using Python fallback (slow / mostly single-core). Run build_windows.bat to enable the multithreaded native path.")

        original_bytes = model.get_bytes()
        glb = _parse_glb(original_bytes)

        supported = []
        skipped_unsupported = 0
        any_work = False

        # First pass: validate each triangle primitive. If every primitive is
        # already closed/manifold, return original bytes exactly and preserve
        # every texture/bake byte with zero serialization round-trip.
        for mesh_index, prim_index, _mesh, primitive in _iter_primitives(glb):
            if not _primitive_is_supported(glb, primitive):
                skipped_unsupported += 1
                continue
            try:
                ok, stats, vertices, faces = _primitive_is_watertight(glb, primitive)
            except GLBError as exc:
                raise GLBError(
                    f"Mesh {mesh_index}, primitive {prim_index} cannot be processed: {exc}"
                ) from exc
            LOGGER.info(
                "[WTiVo Fast Merge] mesh=%d primitive=%d initial_render_topology=%s "
                "vertices=%s faces=%s render_bad_edges=%d render_bad_fans=%d",
                mesh_index, prim_index, ok, f"{len(vertices):,}", f"{len(faces):,}",
                stats["bad_edge_groups"], stats["bad_vertex_fans"],
            )
            if not ok and topology_mode == "TEXTURE_SAFE" and _has_attribute_seams_or_texture(glb, primitive):
                LOGGER.info(
                    "[WTiVo Fast Merge] mesh=%d primitive=%d has render-vertex attribute seams; "
                    "TEXTURE_SAFE mode will preserve them and only weld POSITION for the geometry test.",
                    mesh_index, prim_index,
                )

            if not ok:
                any_work = True
                supported.append((mesh_index, prim_index, primitive, vertices, faces))

        if not any_work:
            LOGGER.info(
                "[WTiVo Fast Merge] Input already watertight. Returning original GLB "
                "unchanged | unsupported_primitives=%d | %.3fs",
                skipped_unsupported, time.perf_counter() - start,
            )
            return (Types.File3D(source=io.BytesIO(original_bytes), file_format="glb"),)

        total_merged = 0
        total_before_v = 0
        total_before_f = 0
        total_after_v = 0
        total_after_f = 0
        native_count = 0
        fallback_count = 0
        final_distances: list[float] = []

        for mesh_index, prim_index, primitive, original_vertices, original_faces in supported:
            before_v = len(original_vertices)
            before_f = len(original_faces)
            total_before_v += before_v
            total_before_f += before_f

            texture_safe = _has_attribute_seams_or_texture(glb, primitive) and topology_mode == "TEXTURE_SAFE"
            solved = False
            last_stats: dict[str, int] = {}
            last_geo_stats: dict[str, int] = {}
            accepted_meta = None
            retry_step = (
                _adaptive_distance_step(distance_step, before_f)
                if adaptive_retry_step else distance_step
            )
            if adaptive_retry_step and retry_step != distance_step:
                LOGGER.info(
                    "[WTiVo Fast Merge] mesh=%d primitive=%d adaptive retry step=%g "
                    "(base=%g, faces=%s)",
                    mesh_index, prim_index, retry_step, distance_step, f"{before_f:,}"
                )

            for attempt in range(1, max_attempts + 1):
                distance = retry_step * attempt
                (
                    new_vertices,
                    new_faces,
                    cluster_map,
                    render_for_original,
                    attr_arrays,
                    used_native,
                ) = _weld_primitive(
                    glb,
                    primitive,
                    original_vertices,
                    original_faces,
                    distance,
                    bool(centroid_merge),
                    bool(remove_degenerate),
                    force_strict=(topology_mode == "STRICT_BLENDER"),
                )
                if used_native:
                    native_count += 1
                else:
                    fallback_count += 1

                # Strict index topology is what Blender/Trimesh sees after GLB
                # import. Texture-safe seams can legitimately keep duplicate
                # render indices at the same XYZ, so also validate geometry by
                # the weld cluster map.
                ok_strict, stats = _validate_watertight(new_faces, len(new_vertices))
                ok_geo, geo_stats = _validate_geometric_watertight(
                    original_faces,
                    len(new_vertices) if render_for_original is None else int(np.max(cluster_map) + 1),
                    cluster_map,
                )
                last_stats = stats
                last_geo_stats = geo_stats

                LOGGER.info(
                    "[WTiVo Fast Merge] mesh=%d primitive=%d try=%d distance=%g "
                    "mode=%s vertices=%s->%s faces=%s->%s "
                    "render_bad_edges=%d render_bad_fans=%d "
                    "geometric_bad_edges=%d geometric_bad_fans=%d "
                    "render_topology=%s geometric_watertight=%s",
                    mesh_index, prim_index, attempt, distance,
                    topology_mode,
                    f"{before_v:,}", f"{len(new_vertices):,}",
                    f"{before_f:,}", f"{len(new_faces):,}",
                    stats["bad_edge_groups"], stats["bad_vertex_fans"],
                    geo_stats["bad_edge_groups"], geo_stats["bad_vertex_fans"],
                    ok_strict, ok_geo,
                )

                # IMPORTANT: for textured/baked GLBs, UV/material/normal seams
                # are allowed to remain duplicated in the glTF render vertex
                # buffer. They must NOT drive the retry loop. The retry loop is
                # driven only by geometric closure after collapsing POSITION
                # clusters. For plain geometry, strict index topology is safe.
                # In TEXTURE_SAFE mode, the positional cluster topology decides the
                # distance search. In STRICT_BLENDER mode, Blender-style indexed
                # topology is the acceptance criterion.
                accepted = ok_geo if topology_mode == "TEXTURE_SAFE" else ok_strict
                if accepted:
                    if texture_safe and not ok_geo:
                        raise AssertionError("Internal error: accepted textured mesh without geometric closure")
                    if render_for_original is None:
                        _rewrite_primitive_attributes(
                            glb,
                            primitive,
                            np.asarray(cluster_map, dtype=np.uint32),
                            np.ascontiguousarray(new_faces, dtype=np.uint32),
                            np.ascontiguousarray(new_vertices, dtype=np.float32),
                            [],
                            before_v,
                        )
                    else:
                        _rewrite_primitive_attributes(
                            glb,
                            primitive,
                            render_for_original,
                            np.ascontiguousarray(new_faces, dtype=np.uint32),
                            np.ascontiguousarray(new_vertices, dtype=np.float32),
                            attr_arrays,
                            before_v,
                        )
                    total_after_v += len(new_vertices)
                    total_after_f += len(new_faces)
                    total_merged += before_v - len(new_vertices)
                    final_distances.append(distance)
                    accepted_meta = (ok_strict, ok_geo, stats, geo_stats)
                    solved = True
                    break

            if not solved:
                last_distance = retry_step * max_attempts
                LOGGER.warning(
                    "[WTiVo Fast Merge] NOT WATERTIGHT | mesh=%d primitive=%d "
                    "after %d attempts | last_distance=%g | "
                    "strict_bad_edges=%d strict_bad_fans=%d | "
                    "geometric_bad_edges=%d geometric_bad_fans=%d | "
                    "RETURNING ORIGINAL GLB UNCHANGED",
                    mesh_index, prim_index, max_attempts, last_distance,
                    last_stats.get("bad_edge_groups", -1),
                    last_stats.get("bad_vertex_fans", -1),
                    last_geo_stats.get("bad_edge_groups", -1),
                    last_geo_stats.get("bad_vertex_fans", -1),
                )
                LOGGER.info(
                    "[WTiVo Fast Merge] No partial result is emitted when watertight "
                    "closure fails; the exact input bytes are returned."
                )
                return (Types.File3D(source=io.BytesIO(original_bytes), file_format="glb"),)

        # Final report deliberately distinguishes the two notions of topology.
        # A texture-safe GLB can have render-vertex seam duplicates even though
        # its welded POSITION topology is closed. This is expected and avoids
        # the destructive UV-stretching behavior of strict glTF index welding.
        if supported:
            LOGGER.info(
                "[WTiVo Fast Merge] topology policy=%s | TEXTURE_SAFE preserves render seams; "
                "STRICT_BLENDER uses merged indexed vertices." % topology_mode
            )

        output_bytes = _rebuild_glb(glb)
        elapsed = time.perf_counter() - start
        max_dist = max(final_distances) if final_distances else 0.0
        LOGGER.info(
            "[WTiVo Fast Merge] SUCCESS | vertices %s->%s | faces %s->%s | merged=%s | "
            "final_distance=%g | native_attempts=%d | python_attempts=%d | %.3fs",
            f"{total_before_v:,}", f"{total_after_v:,}",
            f"{total_before_f:,}", f"{total_after_f:,}", f"{total_merged:,}",
            max_dist, native_count, fallback_count, elapsed,
        )

        return (Types.File3D(source=io.BytesIO(output_bytes), file_format="glb"),)

NODE_CLASS_MAPPINGS = {
    "WTiVoFastMergeByDistance": WTiVoFastMergeByDistance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WTiVoFastMergeByDistance": "WTiVo Fast Merge by Distance",
}
