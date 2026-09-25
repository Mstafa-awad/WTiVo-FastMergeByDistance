from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional

import numpy as np

_DLL_NAMES = (
    "wtivo_fast_weld_mt.dll",
    "wtivo_fast_weld_v7.dll",
    "WTiVoFastWeld_v7.dll",
)
_DLL = None
_THREAD_COUNT = None


def _load_dll() -> Optional[ctypes.CDLL]:
    here = Path(__file__).resolve().parent
    for name in _DLL_NAMES:
        path = here / "native" / name
        if not path.is_file():
            continue
        dll = ctypes.CDLL(str(path))
        dll.wtivo_weld_f32_u32.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
            ctypes.c_float, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)),
        ]
        dll.wtivo_weld_f32_u32.restype = ctypes.c_int
        dll.wtivo_free.argtypes = [ctypes.c_void_p]
        dll.wtivo_free.restype = None
        if hasattr(dll, "wtivo_thread_count"):
            dll.wtivo_thread_count.argtypes = []
            dll.wtivo_thread_count.restype = ctypes.c_int
        return dll
    return None


def native_available() -> bool:
    global _DLL
    if _DLL is None:
        try:
            _DLL = _load_dll() or False
        except OSError:
            _DLL = False
    return bool(_DLL)


def native_thread_count() -> int:
    global _THREAD_COUNT
    if _THREAD_COUNT is not None:
        return int(_THREAD_COUNT)
    if not native_available():
        return 0
    if hasattr(_DLL, "wtivo_thread_count"):
        try:
            _THREAD_COUNT = int(_DLL.wtivo_thread_count())
        except Exception:
            _THREAD_COUNT = 0
    else:
        _THREAD_COUNT = 0
    return int(_THREAD_COUNT)


def native_weld(vertices, faces, distance, centroid_merge, remove_degenerate):
    global _DLL
    if _DLL is None:
        try:
            _DLL = _load_dll() or False
        except OSError:
            _DLL = False
    if not _DLL:
        return None

    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.uint32)
    if len(vertices) > 0xFFFFFFFF:
        raise ValueError("Native weld supports at most 2^32-1 vertices.")

    out_v_ptr = ctypes.POINTER(ctypes.c_float)()
    out_v_count = ctypes.c_uint32(0)
    out_i_ptr = ctypes.POINTER(ctypes.c_uint32)()
    out_i_count = ctypes.c_uint32(0)
    out_rep_ptr = ctypes.POINTER(ctypes.c_uint32)()
    out_remap_ptr = ctypes.POINTER(ctypes.c_uint32)()

    rc = _DLL.wtivo_weld_f32_u32(
        vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(vertices.shape[0]),
        faces.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(faces.size),
        ctypes.c_float(float(distance)),
        ctypes.c_int(1 if centroid_merge else 0),
        ctypes.c_int(1 if remove_degenerate else 0),
        ctypes.byref(out_v_ptr), ctypes.byref(out_v_count),
        ctypes.byref(out_i_ptr), ctypes.byref(out_i_count),
        ctypes.byref(out_rep_ptr), ctypes.byref(out_remap_ptr),
    )
    if rc != 0:
        raise RuntimeError(f"WTiVo native weld failed with error code {rc}.")
    try:
        out_v = np.ctypeslib.as_array(out_v_ptr, shape=(int(out_v_count.value) * 3,)).copy().reshape(-1, 3)
        out_i = np.ctypeslib.as_array(out_i_ptr, shape=(int(out_i_count.value),)).copy().reshape(-1, 3)
        reps = np.ctypeslib.as_array(out_rep_ptr, shape=(int(out_v_count.value),)).copy()
        remap = np.ctypeslib.as_array(out_remap_ptr, shape=(int(vertices.shape[0]),)).copy()
        return (
            np.ascontiguousarray(out_v, dtype=np.float32),
            np.ascontiguousarray(out_i, dtype=np.uint32),
            np.ascontiguousarray(reps, dtype=np.uint32),
            np.ascontiguousarray(remap, dtype=np.uint32),
        )
    finally:
        if out_v_ptr:
            _DLL.wtivo_free(out_v_ptr)
        if out_i_ptr:
            _DLL.wtivo_free(out_i_ptr)
        if out_rep_ptr:
            _DLL.wtivo_free(out_rep_ptr)
        if out_remap_ptr:
            _DLL.wtivo_free(out_remap_ptr)
