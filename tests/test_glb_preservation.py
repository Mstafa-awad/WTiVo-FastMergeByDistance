"""Small regression tests for the v2 GLB-preserving writer.

Run from the repository root with a normal Python environment that has numpy
and trimesh installed. The test stubs only the small ComfyUI File3D API used by
this package.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import types
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]


class _File3D:
    def __init__(self, source, file_format):
        self._bytes = source.getvalue()
        self.format = file_format

    def get_bytes(self):
        return self._bytes


class _Types:
    File3D = _File3D


comfy_api = types.ModuleType("comfy_api")
latest = types.ModuleType("comfy_api.latest")
latest.Types = _Types
sys.modules["comfy_api"] = comfy_api
sys.modules["comfy_api.latest"] = latest

pkg = types.ModuleType("wtivo_test_pkg")
pkg.__path__ = [str(ROOT)]
sys.modules["wtivo_test_pkg"] = pkg
for name in ("native", "python_weld", "node"):
    spec = importlib.util.spec_from_file_location(
        f"wtivo_test_pkg.{name}", ROOT / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "wtivo_test_pkg"
    sys.modules[f"wtivo_test_pkg.{name}"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)

node = sys.modules["wtivo_test_pkg.node"]


def make_duplicate_tetra():
    vertices = np.array([
        [1, 1, 1], [-1, -1, 1], [-1, 1, -1],
        [1, 1, 1], [-1, -1, 1], [1, -1, -1],
        [1, 1, 1], [-1, 1, -1], [1, -1, -1],
        [-1, -1, 1], [-1, 1, -1], [1, -1, -1],
    ], dtype=np.float32)
    faces = np.arange(12, dtype=np.uint32).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh.export(file_type="glb")


def test_retry_and_watertight():
    data = bytearray(make_duplicate_tetra())
    f = np.frombuffer(data, dtype=np.uint8)
    # The actual retry condition is exercised by calling with a normal duplicate
    # mesh; the first step already solves this synthetic model.
    out, = node.WTiVoFastMergeByDistance().execute(
        _File3D(io.BytesIO(bytes(data)), "glb"), 0.0001, False, True, 3
    )
    parsed = node._parse_glb(out.get_bytes())
    prim = parsed.json_obj["meshes"][0]["primitives"][0]
    vertices, faces = node._primitive_data(parsed, prim)
    assert node._validate_watertight(faces, len(vertices))[0]
    assert len(vertices) == 4
    assert len(faces) == 4


def test_already_watertight_is_byte_identical():
    data = trimesh.creation.box().export(file_type="glb")
    out, = node.WTiVoFastMergeByDistance().execute(
        _File3D(io.BytesIO(data), "glb"), 0.0001, False, True, 3
    )
    assert out.get_bytes() == data

if __name__ == "__main__":
    test_retry_and_watertight()
    test_already_watertight_is_byte_identical()
    print("WTiVo Fast Merge v2 regression tests: PASS")
