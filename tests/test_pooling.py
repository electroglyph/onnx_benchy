import numpy as np

from onnx_benchy.pooling import apply_pooling, l2_normalize, pool_and_norm


def _hidden():
    # [batch=2, seq=4, hidden=3]
    h = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    m = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.int64)
    return h, m


def test_cls():
    h, m = _hidden()
    out = apply_pooling(h, m, "cls")
    assert out.shape == (2, 3)
    assert (out[0] == h[0, 0]).all()


def test_mean_masked():
    h, m = _hidden()
    out = apply_pooling(h, m, "mean")
    assert out.shape == (2, 3)
    assert np.allclose(out[0], h[0, :3].mean(axis=0))
    assert np.allclose(out[1], h[1, :2].mean(axis=0))


def test_max_ignores_pads():
    h = np.array([[[1.0, 9.0], [5.0, 2.0], [99.0, 99.0]]])
    m = np.array([[1, 1, 0]])
    out = apply_pooling(h, m, "max")
    assert (out[0] == [5.0, 9.0]).all()


def test_lasttoken():
    h, m = _hidden()
    out = apply_pooling(h, m, "lasttoken")
    assert (out[0] == h[0, 2]).all()
    assert (out[1] == h[1, 1]).all()


def test_none_passthrough_and_rank_check():
    x = np.zeros((2, 5), dtype=np.float32)
    m = np.ones((2, 4), dtype=np.int64)
    assert apply_pooling(x, m, "none").shape == (2, 5)
    try:
        apply_pooling(np.zeros((2, 4, 5), dtype=np.float32), m, "none")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for rank-3 + none")


def test_normalize_unit_norm():
    x = np.array([[3.0, 4.0]])
    n = l2_normalize(x)
    assert np.allclose(np.linalg.norm(n), 1.0)


def test_pool_and_norm():
    h, m = _hidden()
    out = pool_and_norm(h, m, "mean", True)
    assert out.shape == (2, 3)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
