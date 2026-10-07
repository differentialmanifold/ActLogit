"""Numerical regression against MLX-LM's sequential recurrence."""

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.gated_delta import gated_delta_ops  # noqa: E402

from actlogit.mlx_delta import gated_delta_ops_chunked, training_delta_context  # noqa: E402


@pytest.mark.parametrize("length,masked", [(5, False), (65, True), (130, False)])
def test_chunked_outputs_and_all_input_gradients(length, masked):
    mx.random.seed(length)
    q = mx.random.normal((2, length, 2, 16)) * 0.1
    k = mx.random.normal(q.shape)
    k = k / mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    v = mx.random.normal((2, length, 4, 8))
    g = mx.random.uniform(low=0.1, high=0.99, shape=(2, length, 4))
    beta = mx.random.uniform(low=0.1, high=0.95, shape=g.shape)
    state = mx.random.normal((2, 4, 8, 16)) * 0.1
    args = (q, k, v, g, beta, state)
    mask = None if not masked else mx.array([[i % 7 != 0 for i in range(length)]] * 2)
    old = gated_delta_ops(*args, mask=mask)
    new = gated_delta_ops_chunked(*args, mask=mask)
    # Reference returns a readout on masked steps; those outputs are discarded.
    for a, b in zip(old, new, strict=True):
        if a.ndim == 4 and a.shape[1] == length and mask is not None:
            a, b = a * mask[..., None, None], b * mask[..., None, None]
        assert mx.allclose(a, b, atol=3e-6, rtol=3e-5).item()

    def loss(fn):
        def f(*xs):
            y, s = fn(*xs, mask=mask)
            if mask is not None:
                y = y * mask[..., None, None]
            return mx.sum(y * y) / length + mx.sum(s * s) * 0.01

        return f

    old_grad = mx.grad(loss(gated_delta_ops), argnums=list(range(6)))(*args)
    new_grad = mx.grad(loss(gated_delta_ops_chunked), argnums=list(range(6)))(*args)
    for a, b in zip(old_grad, new_grad, strict=True):
        assert mx.all(mx.isfinite(b)).item()
        assert mx.allclose(a, b, atol=4e-6, rtol=5e-4).item()


def test_chunked_collinear_keys_and_extreme_decay():
    mx.random.seed(17)
    for decay in (0.0, 1e-8, 0.999999, 1.0):
        q = mx.ones((1, 67, 1, 16)) / 4
        v = mx.random.normal((1, 67, 1, 8))
        g = mx.full((1, 67, 1), decay)
        beta = mx.full(g.shape, 0.999)
        a = gated_delta_ops(q, q, v, g, beta)
        b = gated_delta_ops_chunked(q, q, v, g, beta)
        for x, y in zip(a, b, strict=True):
            assert mx.all(mx.isfinite(y)).item()
            assert mx.allclose(x, y, atol=8e-4, rtol=8e-4).item()


def test_training_context_restores_fallback_on_error():
    from types import SimpleNamespace

    from mlx_lm.models import gated_delta

    original = gated_delta.gated_delta_ops
    with pytest.raises(RuntimeError):
        with training_delta_context(SimpleNamespace(model_type="qwen3_5"), True):
            assert gated_delta.gated_delta_ops is not original
            raise RuntimeError("test")
    assert gated_delta.gated_delta_ops is original
