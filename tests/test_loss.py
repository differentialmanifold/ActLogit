import pytest
import torch

from actlogit.loss import forward_kl


def test_forward_direction_and_student_gradient():
    logits = torch.tensor([[1.4, -0.2, 0.7]], requires_grad=True)
    q = torch.tensor([[0.05, 0.85, 0.10]], requires_grad=True)
    result = forward_kl(logits, q)
    p = logits.detach().softmax(-1)
    expected = (q.detach() * (q.detach().log() - p.log())).sum()
    reverse = (p * (p.log() - q.detach().log())).sum()
    assert torch.allclose(result.forward_kl.sum(), expected)
    assert not torch.allclose(expected, reverse)
    result.forward_kl.sum().backward()
    assert torch.allclose(logits.grad, p - q.detach(), atol=1e-7)
    assert q.grad is None


def test_soft_ce_has_same_gradient_and_zero_targets_are_safe():
    logits = torch.tensor([[-2.0, 0.0, 3.0]], requires_grad=True)
    result = forward_kl(logits, torch.tensor([[8.0, 2.0, 0.0]]))
    kl_grad = torch.autograd.grad(result.forward_kl.sum(), logits, retain_graph=True)[0]
    ce_grad = torch.autograd.grad(result.cross_entropy.sum(), logits)[0]
    assert torch.isfinite(result.forward_kl).all()
    assert torch.equal(kl_grad, ce_grad)
    assert kl_grad[0, 2] > 0  # Zero-target actions still participate in normalization.


def test_padding_is_excluded_but_low_probability_action_is_retained():
    logits = torch.tensor([[-80.0, 2.0, 9999.0]], requires_grad=True)
    result = forward_kl(
        logits, torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[True, True, False]])
    )
    assert result.forward_kl.item() == pytest.approx(82.0)
    result.forward_kl.sum().backward()
    assert logits.grad[0, 0] == pytest.approx(-1.0)
    assert logits.grad[0, 2] == 0


@pytest.mark.parametrize("target", [[[0.0, 0.0]], [[-1.0, 2.0]], [[float("nan"), 1.0]]])
def test_invalid_targets_fail(target):
    with pytest.raises(ValueError):
        forward_kl(torch.zeros(1, 2), torch.tensor(target))


def test_illegal_actions_cannot_have_target_mass():
    with pytest.raises(ValueError, match="padded"):
        forward_kl(torch.zeros(1, 2), torch.ones(1, 2), torch.tensor([[True, False]]))
