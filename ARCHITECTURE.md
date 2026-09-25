# Why the two topology modes exist

Standard glTF uses one index stream for a primitive; POSITION, UV, normals,
tangents, colors, joints and weights are all vertex attributes selected by that
same index. Blender therefore exports UV/normal discontinuities as separate
glTF render vertices.

`STRICT_BLENDER` welds those indexed vertices by position, like Merge by
Distance, so Blender sees shared indices. At a UV seam this can change which
UV value a triangle corner receives.

`TEXTURE_SAFE` first computes positional weld clusters, then rebuilds render
vertices keyed by `(welded position cluster + original attributes)`. This keeps
UV seams and baked textures intact, but it necessarily allows multiple glTF
render vertices at the same XYZ where attributes differ.

An exact one-index/multiple-UV representation cannot be encoded in core glTF.
A real solution for both goals requires either changing/rebaking the texture
mapping or retaining Blender's face-corner UV representation outside standard
glTF.
