import pytest
import torch

from pytorch_metric_learning.losses import VICRegLoss


def views(dtype, scale):
    first = (
        torch.tensor([[2.0, 2.0], [-2.0, -2.0], [1.0, 1.0], [-1.0, -1.0]], dtype=dtype)
        * scale
    )
    second = (first * 0.75 + 0.125).to(dtype)
    return first.requires_grad_(), second.requires_grad_()


def reference_loss(first, second, weights):
    n, dimensions = first.shape
    invariance = ((first - second) ** 2).sum() / (n * dimensions)
    variance = first.new_zeros(())
    covariance = first.new_zeros(())
    for values in (first, second):
        centered = values - values.sum(dim=0) / n
        variances = (centered**2).sum(dim=0) / (n - 1)
        variance = variance + (1 - (variances + 1e-4).sqrt()).clamp_min(0).sum() / (
            2 * dimensions
        )
        for i in range(dimensions):
            for j in range(dimensions):
                if i != j:
                    entry = (centered[:, i] * centered[:, j]).sum() / (n - 1)
                    covariance = covariance + entry**2 / dimensions
    return weights[0] * invariance + weights[1] * variance + weights[2] * covariance


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [0.05, 10.0])
@pytest.mark.parametrize("weights", [(25, 25, 1), (2, 3, 0.5)])
def test_low_precision_views_match_the_double_objective_and_gradients(
    dtype, scale, weights
):
    first, second = views(dtype, scale)
    ref_first = first.detach().double().requires_grad_()
    ref_second = second.detach().double().requires_grad_()
    expected = reference_loss(ref_first, ref_second, weights)
    expected_gradients = torch.autograd.grad(expected, (ref_first, ref_second))
    actual = VICRegLoss(*weights)(first, second)
    actual_gradients = torch.autograd.grad(actual, (first, second))
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-6)
    for gradient, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, reference.to(dtype), rtol=0.01, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_autocast_does_not_narrow_covariance_statistics(dtype):
    first, second = views(torch.float32, 100.0)
    ref_first = first.detach().double().requires_grad_()
    ref_second = second.detach().double().requires_grad_()
    expected = reference_loss(ref_first, ref_second, (25, 25, 1))
    expected_gradients = torch.autograd.grad(expected, (ref_first, ref_second))
    with torch.autocast("cpu", dtype=dtype):
        actual = VICRegLoss()(first, second)
        assert torch.is_autocast_enabled("cpu")
    gradients = torch.autograd.grad(actual, (first, second))
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-6)
    for gradient, reference in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(gradient.double(), reference, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_precision_views_keep_their_dtype_and_values(dtype):
    first, second = views(dtype, 0.2)
    actual = VICRegLoss()(first, second)
    expected = reference_loss(first.double(), second.double(), (25, 25, 1))
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-6)


def test_a_double_precision_view_is_not_narrowed_by_a_half_precision_view():
    first, second = views(torch.float16, 10.0)
    second = second.detach().double().requires_grad_()
    actual = VICRegLoss()(first, second)
    expected = reference_loss(first.double(), second, (25, 25, 1))
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected)
