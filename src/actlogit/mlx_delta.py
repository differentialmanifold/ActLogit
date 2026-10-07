"""Training-only chunked gated-delta recurrence, vendored under the MIT license.

Source: https://github.com/ml-explore/mlx-lm/pull/1389 (not merged upstream).
Pinned revision: 6fc3a295717fcf8f15f72f9425192c731044c8aa, tsato081/mlx-lm.
GQA optimization credited upstream to SudarkinV. Inference is never patched.
Local change: solve blocks are 8 (upstream 16), reducing error for repeated keys.
"""

# MIT License
#
# Copyright © 2023 Apple Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from contextlib import contextmanager

import mlx.core as mx

CHUNK_SIZE = 64

# Solve block size; bounds fp32 error growth when keys repeat in a chunk.
SUB_BLOCK = 8


def _solve_strict_lower(A: mx.array, b: mx.array, sb: int = SUB_BLOCK) -> mx.array:
    """Solve (I - A) x = b for strictly lower-triangular (nilpotent) A.

    Blocked forward substitution; a global Neumann expansion can
    overflow fp32 when keys repeat within a chunk.
    """
    C = A.shape[-1]

    def doubling(Aii, rhs, n):
        x = rhs
        if n <= 1:
            return x
        P = Aii
        steps = (n - 1).bit_length()
        for s in range(steps):
            x = x + P @ x
            if s != steps - 1:
                P = P @ P
        return x

    if C <= sb:
        return doubling(A, b, C)

    nb = (C + sb - 1) // sb
    blocks = []
    for i in range(nb):
        lo, hi = i * sb, min((i + 1) * sb, C)
        rhs = b[..., lo:hi, :]
        if i > 0:
            prev = blocks[0] if i == 1 else mx.concatenate(blocks, axis=-2)
            rhs = rhs + A[..., lo:hi, :lo] @ prev
        blocks.append(doubling(A[..., lo:hi, lo:hi], rhs, hi - lo))
    return mx.concatenate(blocks, axis=-2)


def _gated_delta_chunk(
    state: mx.array,  # [B, Hv, Dk, Dv]
    q: mx.array,  # [B, Hk, C, Dk]
    k: mx.array,  # [B, Hk, C, Dk]
    v: mx.array,  # [B, Hv, C, Dv]
    g: mx.array,  # [B, Hv, C], gating in (0, 1)
    beta: mx.array,  # [B, Hv, C]
    repeat_factor: int = 1,
) -> tuple[mx.array, mx.array]:
    """Run C timesteps as one triangular solve (gated UT/WY transform).

    Exact reformulation of the sequential recurrence; runs in fp32.

    Under grouped-query attention (repeat_factor > 1) the C x C key Gram
    and q . k^T products depend only on the Hk query/key heads, so they
    are formed before the GQA broadcast to Hv; the per-Hv gating is
    applied afterwards. This avoids materializing the repeated q/k for the
    whole sequence and shrinks those two matmuls by repeat_factor.
    """
    C = q.shape[2]
    Hk = q.shape[1]
    Hv = v.shape[1]
    orig_dtype = q.dtype

    q = q.astype(mx.float32)
    k = k.astype(mx.float32)
    v = v.astype(mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    state = state.astype(mx.float32)

    # Log-domain cumulative decay; the clamp keeps -inf out of the cumsum.
    g_log = mx.log(mx.maximum(g, 1e-12))  # [B, H, C]
    g_cumlog = mx.cumsum(g_log, axis=-1)
    g_last = g_cumlog[..., -1:]

    # Zero the upper triangle before exp: it overflows, and inf * 0 = NaN.
    tril_ones = mx.tril(mx.ones((C, C), dtype=mx.float32))
    L_diff = (g_cumlog[..., :, None] - g_cumlog[..., None, :]) * tril_ones
    L_mask = mx.exp(L_diff) * tril_ones

    # GQA sharing: the C x C products only need the Hk heads. Form them
    # first, then broadcast q/k and the products to Hv.
    kkT = k @ mx.swapaxes(k, -1, -2)  # [B, Hk, C, C], kkT[i, j] = <k_i, k_j>
    qkT = q @ mx.swapaxes(k, -1, -2)  # [B, Hk, C, C]
    if repeat_factor > 1:
        B = q.shape[0]

        def _to_hv(x):  # [B, Hk, C, D] -> [B, Hv, C, D]
            D = x.shape[-1]
            return mx.broadcast_to(x[:, :, None], (B, Hk, repeat_factor, C, D)).reshape(B, Hv, C, D)

        kkT = _to_hv(kkT)
        qkT = _to_hv(qkT)
        q = _to_hv(q)
        k = _to_hv(k)

    v_beta = v * beta[..., None]  # [B, Hv, C, Dv]
    k_beta = k * beta[..., None]  # [B, Hv, C, Dk]

    # (k_beta @ k^T)[i, j] = beta_i * <k_i, k_j> = beta_i * kkT[i, j]
    strict_lower = mx.tril(mx.ones((C, C), dtype=mx.float32), k=-1)
    A = -(beta[..., :, None] * kkT) * L_mask * strict_lower

    decay_exp = mx.exp(g_cumlog)[..., None]  # [B, Hv, C, 1]
    rhs = mx.concatenate([v_beta, k_beta * decay_exp], axis=-1)
    sol = _solve_strict_lower(A, rhs)
    v_corrected, k_cumdecay = mx.split(sol, [v.shape[-1]], axis=-1)

    v_new = v_corrected - k_cumdecay @ state  # [B, Hv, C, Dv]
    y_inter = (q * decay_exp) @ state  # [B, Hv, C, Dv]

    attn = qkT * L_mask
    y = y_inter + attn @ v_new

    state_decay = mx.exp(g_last)[..., None]
    decay_to_end = mx.exp(g_last - g_cumlog)[..., None]
    new_state = state * state_decay + mx.swapaxes(k * decay_to_end, -1, -2) @ v_new

    # The state stays fp32 across chunks; casting at boundaries drifts.
    return y.astype(orig_dtype), new_state


_gated_delta_chunk_checkpointed = mx.checkpoint(_gated_delta_chunk)


def gated_delta_ops_chunked(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
    mask: mx.array | None = None,
    chunk_size: int | None = None,
) -> tuple[mx.array, mx.array]:
    """
    Chunk-parallel implementation of prompt prefill for scalar gating.

    Equivalent to gated_delta_ops but processes chunk_size timesteps at a
    time with dense matmuls, each chunk wrapped in mx.checkpoint, so the
    autodiff graph is O(T / chunk_size) instead of O(T).

    Shapes:
      - q, k: [B, T, Hk, Dk]
      - v: [B, T, Hv, Dv]
      - g: [B, T, Hv] (scalar gating only, values in (0, 1))
      - beta: [B, T, Hv]
      - state: [B, Hv, Dv, Dk]
    Returns:
      - y: [B, T, Hv, Dv]
      - state: [B, Hv, Dv, Dk]
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    C = chunk_size or CHUNK_SIZE
    repeat_factor = Hv // Hk

    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    # q/k stay on the Hk query/key heads; the chunk shares their C x C
    # products across the GQA group and broadcasts to Hv internally.

    # Masked steps are identities on the state (g = 1, beta = 0).
    if mask is not None:
        m = mask[..., None]
        g = mx.where(m, g, mx.ones_like(g))
        beta = beta * m

    # Pad T to a multiple of C with identity steps (g = 1, beta = 0).
    pad_len = (C - (T % C)) % C
    if pad_len > 0:
        q = mx.pad(q, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        k = mx.pad(k, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        g = mx.concatenate([g, mx.ones((B, pad_len, Hv), dtype=g.dtype)], axis=1)
        beta = mx.pad(beta, [(0, 0), (0, pad_len), (0, 0)])

    num_chunks = (T + pad_len) // C

    # [B, T, H, D] -> [B, H, Nc, C, D]; q/k keep their Hk heads.
    q = mx.swapaxes(q, 1, 2).reshape(B, Hk, num_chunks, C, Dk)
    k = mx.swapaxes(k, 1, 2).reshape(B, Hk, num_chunks, C, Dk)
    v = mx.swapaxes(v, 1, 2).reshape(B, Hv, num_chunks, C, Dv)
    g = mx.swapaxes(g, 1, 2).reshape(B, Hv, num_chunks, C)
    beta = mx.swapaxes(beta, 1, 2).reshape(B, Hv, num_chunks, C)

    # [B, Hv, Dv, Dk] -> [B, Hv, Dk, Dv]
    state = mx.swapaxes(state.astype(mx.float32), -1, -2)

    ys = []
    for ci in range(num_chunks):
        y_c, state = _gated_delta_chunk_checkpointed(
            state,
            q[:, :, ci],
            k[:, :, ci],
            v[:, :, ci],
            g[:, :, ci],
            beta[:, :, ci],
            repeat_factor,
        )
        ys.append(y_c)

    y = mx.concatenate(ys, axis=2)
    if pad_len > 0:
        y = y[:, :, :T, :]
    y = mx.swapaxes(y, 1, 2)

    return y, mx.swapaxes(state, -1, -2)


@contextmanager
def training_delta_context(model, enabled=False):
    """Scope the opt-in fallback to this standalone training operation.

    MLX-LM's model calls a module-level recurrence. Restore it even on failure;
    the Metal inference path does not call this fallback. Do not concurrently
    train a second model in this process while this context is active.
    """
    if not enabled:
        yield
        return
    if getattr(model, "model_type", None) not in {"qwen3_5", "qwen3_5_moe"}:
        raise ValueError("mlx_chunked_gated_delta currently supports Qwen3.5 only")
    from mlx_lm.models import gated_delta

    original = gated_delta.gated_delta_ops

    def dispatch(q, k, v, g, beta, state=None, mask=None):
        if g.ndim != 3 or q.shape[1] == 1:
            return original(q, k, v, g, beta, state, mask)
        return gated_delta_ops_chunked(q, k, v, g, beta, state, mask)

    gated_delta.gated_delta_ops = dispatch
    try:
        yield
    finally:
        gated_delta.gated_delta_ops = original
