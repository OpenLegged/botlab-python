"""quat_multiply must match both pre-refactor copies bit for bit."""

import numpy as np

from botlab._runtime._quat import quat_multiply


def _old_vectorised(a, b):
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )


def _old_scalar(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def test_batched_matches_old_vectorised_exactly():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(512, 4)) * rng.choice([1e-3, 1, 1e3], size=(512, 1))
    b = rng.normal(size=(512, 4))
    assert np.array_equal(quat_multiply(a, b), _old_vectorised(a, b))


def test_single_matches_old_scalar_exactly():
    rng = np.random.default_rng(1)
    for _ in range(500):
        a = rng.normal(size=4)
        b = rng.normal(size=4)
        got = quat_multiply(a, b)
        assert got.shape == (4,)
        assert np.array_equal(got, np.asarray(_old_scalar(tuple(a), tuple(b))))
