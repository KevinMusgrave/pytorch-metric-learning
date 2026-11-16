import math
from decimal import Decimal, localcontext

import pytest
import torch

from pytorch_metric_learning import distances

from .. import TEST_DEVICE


def decimal_normalization(values, weights, p):
    with localcontext(prec=100):
        values = [Decimal.from_float(value) for value in values]
        weights = [Decimal.from_float(value) for value in weights]
        if p == math.inf:
            norm = max(abs(value) for value in values)
            derivative = [Decimal(int(abs(value) == norm)) for value in values]
        else:
            norm = sum((abs(value) ** p for value in values), start=Decimal(0)) ** (
                Decimal(1) / p
            )
            derivative = [value ** (p - 1) / norm ** (p - 1) for value in values]
        output = [value / norm for value in values]
        dot = sum(
            (weight * value for weight, value in zip(weights, output, strict=True)),
            start=Decimal(0),
        )
        gradient = [
            (weight - dot * direction) / norm
            for weight, direction in zip(weights, derivative, strict=True)
        ]
        return [float(value) for value in output], [float(value) for value in gradient]


@pytest.mark.parametrize(
    "dtype,scale",
    [
        (torch.float16, 10000.0),
        (torch.bfloat16, 1e20),
        (torch.float32, 1e20),
        (torch.float64, 1e200),
    ],
)
@pytest.mark.parametrize("p", [1, 2, 3, math.inf])
@pytest.mark.parametrize("dim", [0, 1])
def test_normalize_large_finite_vectors_preserves_values_and_gradients(
    dtype, scale, p, dim
):
    vector = torch.tensor([3 * scale, 4 * scale], device=TEST_DEVICE, dtype=dtype)
    query = (vector[:, None] if dim == 0 else vector[None, :]).clone().requires_grad_()
    weights = query.new_tensor([0.75, -0.25]).reshape(query.shape)
    actual = distances.LpDistance(p=p).normalize(query, dim=dim)
    gradient = torch.autograd.grad((actual * weights).sum(), query)[0]
    expected, expected_gradient = decimal_normalization(
        vector.double().cpu().tolist(), [0.75, -0.25], p
    )
    tolerance = 5e-3 if dtype in (torch.float16, torch.bfloat16) else 3e-6
    torch.testing.assert_close(
        actual, actual.new_tensor(expected).reshape(query.shape), rtol=tolerance, atol=0
    )
    torch.testing.assert_close(
        gradient,
        gradient.new_tensor(expected_gradient).reshape(query.shape),
        rtol=tolerance,
        atol=0,
    )
    assert actual.dtype == dtype


@pytest.mark.parametrize(
    "dtype,scale",
    [(torch.bfloat16, 1e20), (torch.float32, 1e20), (torch.float64, 1e200)],
)
def test_large_cosine_vectors_reach_matrix_and_pairwise_distance_paths(dtype, scale):
    query = torch.tensor(
        [[3 * scale, 4 * scale]], device=TEST_DEVICE, dtype=dtype, requires_grad=True
    )
    reference = torch.tensor([[1.0, 0.0]], device=TEST_DEVICE, dtype=dtype)
    norm = math.hypot(*query.detach().double().cpu().flatten().tolist())
    expected = float(query.detach()[0, 0]) / norm
    metric = distances.CosineSimilarity()
    pairwise = metric.pairwise(query, reference)
    matrix = metric(query, reference)
    torch.testing.assert_close(
        pairwise, pairwise.new_tensor([expected]), rtol=5e-3, atol=0
    )
    torch.testing.assert_close(matrix.flatten(), pairwise)
    gradient = torch.autograd.grad(pairwise.sum(), query)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() > 0


@pytest.mark.parametrize("shape", [(0, 3), (2, 0)])
def test_empty_normalization_keeps_shape_and_graph(shape):
    query = torch.empty(shape, device=TEST_DEVICE, requires_grad=True)
    actual = distances.LpDistance().normalize(query)
    actual.sum().backward()
    assert actual.shape == query.shape
    torch.testing.assert_close(query.grad, torch.zeros_like(query))


def test_zero_norm_order_retains_count_normalization():
    query = torch.tensor([[2.0, 0.0, 4.0]], device=TEST_DEVICE, requires_grad=True)
    actual = distances.LpDistance(p=0).normalize(query)
    torch.testing.assert_close(actual, query / 2)
    actual.sum().backward()
    torch.testing.assert_close(query.grad, torch.full_like(query, 0.5))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("p", [1, 2, 3, math.inf])
def test_normal_range_keeps_existing_values_and_gradients_exactly(dtype, p):
    query = torch.tensor(
        [[0.2, 0.4, 0.6], [0.1, -0.3, 0.7], [0.0, 0.0, 0.0]],
        device=TEST_DEVICE,
        dtype=dtype,
        requires_grad=True,
    )
    reference = query.detach().clone().requires_grad_()
    work = reference.float() if dtype in (torch.float16, torch.bfloat16) else reference
    expected = torch.nn.functional.normalize(
        work, p=p, dim=1, eps=max(1e-12, torch.finfo(dtype).tiny)
    ).to(dtype)
    actual = distances.LpDistance(p=p).normalize(query)
    weights = query.new_tensor([[0.3, -0.4, 0.5], [0.6, 0.7, -0.8], [0.2, 0.3, 0.4]])
    actual_gradient = torch.autograd.grad((actual * weights).sum(), query)[0]
    expected_gradient = torch.autograd.grad((expected * weights).sum(), reference)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,scale", [(torch.float32, 1e20), (torch.float64, 1e200)])
def test_mixed_batch_scales_only_the_large_vector(dtype, scale):
    query = torch.tensor(
        [[3 * scale, 4 * scale], [0.3, 0.4]],
        device=TEST_DEVICE,
        dtype=dtype,
        requires_grad=True,
    )
    ordinary = query.detach()[1:].clone().requires_grad_()
    actual = distances.LpDistance().normalize(query)
    expected = torch.nn.functional.normalize(ordinary, dim=1)
    torch.testing.assert_close(actual[1:], expected, rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(actual[1:].sum(), query)[0][1:]
    expected_gradient = torch.autograd.grad(expected.sum(), ordinary)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    torch.testing.assert_close(actual[0], query.new_tensor([0.6, 0.8]))
