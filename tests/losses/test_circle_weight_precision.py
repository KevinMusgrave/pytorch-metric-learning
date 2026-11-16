import math

import pytest
import torch

from pytorch_metric_learning import losses

from .. import TEST_DEVICE


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("gamma", [80.0, 256.0, 1e5])
@pytest.mark.parametrize("margin", [0.4, 0.5])
def test_circle_finite_adaptive_weights_preserve_values_and_gradients(
    dtype, gamma, margin
):
    matrix = torch.full(
        (4, 4), 0.5, device=TEST_DEVICE, dtype=dtype, requires_grad=True
    )
    labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
    actual = losses.CircleLoss(margin=margin, gamma=gamma)(
        torch.zeros(4, 2, device=TEST_DEVICE, dtype=dtype), labels, mat=matrix
    )
    gradient = torch.autograd.grad(actual, matrix)[0]
    alpha_p, alpha_n = 1 + margin - 0.5, 0.5 + margin
    total_logit = (
        -gamma * alpha_p * (0.5 - (1 - margin))
        + gamma * alpha_n * (0.5 - margin)
        + math.log(2)
    )
    expected = total_logit + math.log1p(math.exp(-total_logit))
    probability = 1 / (1 + math.exp(-total_logit))
    expected_gradient = torch.zeros_like(matrix)
    same = labels[:, None] == labels[None, :]
    positive = same & ~torch.eye(4, device=TEST_DEVICE, dtype=torch.bool)
    expected_gradient[positive] = -gamma * alpha_p * probability / 4
    expected_gradient[~same] = gamma * alpha_n * probability / 8
    tolerance = 5e-3 if dtype in (torch.float16, torch.bfloat16) else 3e-6
    torch.testing.assert_close(
        actual, actual.new_tensor(expected), rtol=tolerance, atol=0
    )
    torch.testing.assert_close(gradient, expected_gradient, rtol=tolerance, atol=0)
    assert actual.dtype == dtype


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_circle_rectangular_references_match_independent_pair_equations(dtype):
    values = [[0.3, 0.5, 0.4, 0.1], [0.2, 0.3, 0.6, 0.7]]
    matrix = torch.tensor(values, device=TEST_DEVICE, dtype=dtype, requires_grad=True)
    labels = torch.tensor([0, 1], device=TEST_DEVICE)
    ref_labels = torch.tensor([0, 0, 1, 2], device=TEST_DEVICE)
    actual = losses.CircleLoss()(
        torch.zeros(2, 2, device=TEST_DEVICE, dtype=dtype),
        labels,
        ref_emb=torch.zeros(4, 2, device=TEST_DEVICE, dtype=dtype),
        ref_labels=ref_labels,
        mat=matrix,
    )
    gradient = torch.autograd.grad(actual, matrix)[0]
    reference = matrix.detach().double().requires_grad_()
    values = []
    for row, label in enumerate(labels):
        positive = reference[row, ref_labels == label]
        negative = reference[row, ref_labels != label]
        attraction = -80 * (1.4 - positive).clamp_min(0).detach() * (positive - 0.6)
        repulsion = 80 * (negative + 0.4).clamp_min(0).detach() * (negative - 0.4)
        values.append(
            torch.nn.functional.softplus(
                torch.logsumexp(attraction, 0) + torch.logsumexp(repulsion, 0)
            )
        )
    expected = torch.stack(values).mean()
    expected_gradient = torch.autograd.grad(expected, reference)[0]
    tolerance = 5e-3 if dtype in (torch.float16, torch.bfloat16) else 3e-6
    torch.testing.assert_close(actual, expected.to(dtype), rtol=tolerance, atol=1e-6)
    torch.testing.assert_close(
        gradient, expected_gradient.to(dtype), rtol=tolerance, atol=1e-6
    )


@pytest.mark.parametrize("labels", [[0, 0, 0], [0, 1, 2]])
def test_circle_missing_pair_category_keeps_differentiable_zero(labels):
    query = torch.eye(3, device=TEST_DEVICE, dtype=torch.float16, requires_grad=True)
    actual = losses.CircleLoss(margin=0.5, gamma=1e5)(
        query, torch.tensor(labels, device=TEST_DEVICE)
    )
    actual.backward()
    assert actual.item() == 0
    torch.testing.assert_close(query.grad, torch.zeros_like(query))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_circle_opposing_large_partitions_retain_small_objective(dtype):
    gamma = 1e8
    matrix = torch.full((4, 4), 0.875, device=TEST_DEVICE, dtype=dtype)
    matrix.fill_diagonal_(1)
    matrix.requires_grad_()
    labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
    actual = losses.CircleLoss(margin=0.625, gamma=gamma)(
        torch.zeros(4, 2, device=TEST_DEVICE, dtype=dtype), labels, mat=matrix
    )
    gradient = torch.autograd.grad(actual, matrix)[0]
    same = labels[:, None] == labels[None, :]
    expected_gradient = torch.zeros_like(matrix)
    expected_gradient[same & ~torch.eye(4, device=TEST_DEVICE, dtype=torch.bool)] = (
        -gamma / 8
    )
    expected_gradient[~same] = gamma / 8
    torch.testing.assert_close(
        actual, actual.new_tensor(math.log(3)), rtol=3e-6, atol=0
    )
    torch.testing.assert_close(gradient, expected_gradient, rtol=3e-6, atol=0)
