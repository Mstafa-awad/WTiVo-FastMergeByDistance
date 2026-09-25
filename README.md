# WTiVo Fast Merge by Distance v7

Fast CPU vertex welding for ComfyUI GLB meshes after QMesh/decimation.

## v7: important changes

- The native C++ core now uses **OpenMP multi-threading** for the expensive spatial-hash candidate search.
- The node reports the detected native thread count.
- The native DLL is loaded from `native/wtivo_fast_weld_mt.dll`.
- `build_windows.bat` builds the DLL with MSVC `/openmp` and x64.
- The adaptive distance sequence remains `step, 2*step, 3*step, ...`.
- The node supports two explicit topology policies: `TEXTURE_SAFE` and `STRICT_BLENDER`.
- `TEXTURE_SAFE` preserves glTF UV/texture/normal/tangent/color/skin/morph attributes but does **not** pretend that a UV seam is the same indexed vertex.
- `STRICT_BLENDER` creates one render vertex per positional weld cluster, matching the indexed topology you get from Blender Merge by Distance, but arbitrary UV seams cannot be preserved in standard glTF without changing/rebaking the texture mapping.

## The glTF limitation

glTF uses one index stream for a primitive and vertex attributes such as POSITION and TEXCOORD_0 are attached to the indexed vertex. Blender's documentation explicitly notes that discontinuous UVs/normals cause separate glTF vertices on export. Therefore an exact combination of:

1. one shared glTF vertex index at a UV seam, and
2. two different UV coordinates at that same vertex, and
3. unchanged texture appearance

cannot be represented directly in standard glTF 2.0.

This is why v6 could preserve the texture but still show seam-separated vertices in Blender, while strict welding could get 0 indexed boundary/non-manifold edges but damage the seam UV.

## Which mode to use

`TEXTURE_SAFE` is the safe choice for already textured/baked game assets. It keeps the original render attributes.

`STRICT_BLENDER` is the diagnostic/geometry-first choice when you need the indexed topology to behave like Merge by Distance. Expect UV seam changes unless the affected vertices already have compatible attributes.

The node does not claim that TEXTURE_SAFE makes a GLB pass Blender's strict Edit Mode non-manifold selection after import. That would be misleading because of the glTF representation above.

## Build

Run `build_windows.bat` from a Developer Command Prompt or a normal shell with CMake and Visual Studio available. The result is copied to `native/wtivo_fast_weld_mt.dll`.

When the DLL is loaded, the log reports something like:

`[WTiVo Fast Merge] Native C++ enabled | OpenMP threads=16`

Without the DLL, the Python fallback is used and will be slower.

## License

This project's original source is MIT licensed and commercially usable. See `LICENSE`.
