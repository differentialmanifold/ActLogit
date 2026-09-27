"""Forward KL on a finite action space, with detached external targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class DistributionLoss:
    forward_kl: Tensor
    cross_entropy: Tensor
    target_entropy: Tensor


def forward_kl(
    logits: Tensor,
    target_weights: Tensor,
    mask: Tensor | None = None,
) -> DistributionLoss:
    """Return per-example KL(q || p), CE(q,p), H(q); q never receives gradients.

    Both distributions cover ALL valid actions. There is no vocabulary or target
    top-k filtering. Padded actions are excluded from both distributions.
    """
    if logits.ndim != 2 or target_weights.shape != logits.shape:
        raise ValueError("logits and targets must have matching [batch, actions] shapes")
    if mask is None:
        mask = torch.ones_like(logits, dtype=torch.bool)
    if mask.shape != logits.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be a boolean tensor with the same shape as logits")
    if not mask.any(dim=-1).all():
        raise ValueError("each example must contain at least one legal action")
    if not torch.isfinite(logits[mask]).all():
        raise ValueError("legal action logits must be finite")
    q = target_weights.detach().to(device=logits.device, dtype=torch.float32)
    if not torch.isfinite(q).all() or (q < 0).any():
        raise ValueError("target weights must be finite and nonnegative")
    if (q[~mask] != 0).any():
        raise ValueError("padded actions must have zero target weight")
    q = q.masked_fill(~mask, 0)
    totals = q.sum(dim=-1, keepdim=True)
    if not torch.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("each target must have a finite positive total weight")
    q = q / totals
    log_p = F.log_softmax(logits.float().masked_fill(~mask, -torch.inf), dim=-1)
    # Avoid 0 * -inf at padding, while retaining all valid actions in the softmax.
    log_p = log_p.masked_fill(~mask, 0)
    cross_entropy = -(q * log_p).sum(dim=-1)
    entropy = -torch.special.xlogy(q, q).sum(dim=-1)
    return DistributionLoss(cross_entropy - entropy, cross_entropy, entropy)
