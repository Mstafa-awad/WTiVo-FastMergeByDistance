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
    def __init__(self, source=None, file_format="glb"):
        self._bytes = source.getvalue() if hasattr(source, "getvalue") else source
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

pkg = types.ModuleType("wtivo_test_pkg_v5")
pkg.__path__ = [str(ROOT)]
sys.modules["wtivo_test_pkg_v5"] = pkg
for name in ("native", "python_weld", "node"):
    spec = importlib.util.spec_from_file_location(
        f"wtivo_test_pkg_v5.{name}", ROOT / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "wtivo_test_pkg_v5"
    sys.modules[f"wtivo_test_pkg_v5.{name}"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)

node = sys.modules["wtivo_test_pkg_v5.node"]


def make_seam_gap_glb():
    # Closed tetrahedron, except one face uses a duplicate copy of vertex 0
    # offset slightly. That duplicate also has a deliberately different UV.
    vertices = np.array([
        [1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1],
        [1.00015, 1, 1],
    ], dtype=np.float32)
    faces = np.array([
        [4, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]
    ], dtype=np.uint32)
    uv = np.array([
        [0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2], [0.9, 0.9]
    ], dtype=np.float32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv)
    return mesh.export(file_type="glb")


def test_uv_is_preserved_when_position_is_welded():
    data = make_seam_gap_glb()
    out, = node.WTiVoFastMergeByDistance().execute(
        _File3D(source=io.BytesIO(data)), 0.0001, False, True, 3
    )
    glb = node._parse_glb(out.get_bytes())
    prim = glb.json_obj["meshes"][0]["primitives"][0]
    vertices, faces = node._primitive_data(glb, prim)
    uv = node._decode_accessor(glb, int(prim["attributes"]["TEXCOORD_0"]))

    # Both UV values survive, and the two seam render vertices now share XYZ.
    seam = np.where(
        np.isclose(vertices[:, 0], 1.0, atol=1e-6)
        & np.isclose(vertices[:, 1], 1.0, atol=1e-6)
        & np.isclose(vertices[:, 2], 1.0, atol=1e-6)
    )[0]
    assert len(seam) == 2
    assert len(np.unique(np.round(uv[seam], 4), axis=0)) == 2

    # The geometric topology is closed when measured through the welded
    # POSITION clusters, while strict indexed topology remains split at the UV seam.
    _sv, _sf, _sr, cluster = node.python_weld(vertices, faces, 1e-7, False, True)
    geo_ok, _ = node._validate_geometric_watertight(
        faces, int(np.max(cluster) + 1), cluster
    )
    assert geo_ok


if __name__ == "__main__":
    test_uv_is_preserved_when_position_is_welded()
    print("WTiVo texture-safe regression test: PASS")
