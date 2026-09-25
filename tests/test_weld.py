import numpy as np

from python_weld import python_weld


def main():
    # Two triangles with duplicated vertices at identical positions.
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)

    out_v, out_f, reps = python_weld(vertices, faces, 1e-5, False, True)
    assert out_v.shape[0] == 3, out_v
    assert out_f.shape[0] == 2, out_f
    assert reps.shape[0] == 3
    print("PASS: duplicate vertex weld")


if __name__ == "__main__":
    main()
