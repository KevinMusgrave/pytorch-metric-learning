import math

import pytest
import torch
import torch.nn.functional as F

from pytorch_metric_learning import distances, losses

from .. import TEST_DEVICE


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("temperature", [1e-5, 0.1, -1e-5])
def test_soft_triple_equal_centers_keep_finite_values_and_optimizer_gradients(
    dtype, temperature
):
    model = losses.SoftTripleLoss(
        2, 2, centers_per_class=2, gamma=temperature, la=2, margin=0
    ).to(device=TEST_DEVICE, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(model.weight.new_tensor([[1, 0]] * 4))
    query = torch.tensor(
        [[1.0, 1.0]], device=TEST_DEVICE, dtype=dtype, requires_grad=True
    )
    logits = model.get_logits(query)
    assert logits.dtype == dtype
    tolerance = (
        2e-2 if dtype == torch.bfloat16 else 3e-3 if dtype == torch.float16 else 3e-6
    )
    torch.testing.assert_close(
        logits, logits.new_full((1, 2), 1 / math.sqrt(2)), rtol=tolerance, atol=0
    )
    value = model(query, torch.tensor([0], device=TEST_DEVICE))
    torch.testing.assert_close(
        value, value.new_tensor(math.log(2)), rtol=tolerance, atol=0
    )
    value.backward()
    torch.testing.assert_close(query.grad, torch.zeros_like(query), rtol=0, atol=1e-6)
    expected = model.weight.new_tensor(
        [[0, -1 / (2 * math.sqrt(2))]] * 2 + [[0, 1 / (2 * math.sqrt(2))]] * 2
    )
    torch.testing.assert_close(model.weight.grad, expected, rtol=tolerance, atol=0)
    previous = model.weight.detach().clone()
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    assert torch.isfinite(model.weight).all()
    assert not torch.equal(previous, model.weight)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("temperature", [1e-5, 0.1, -1e-5])
def test_soft_triple_class_weighting_matches_float64_objective_and_gradients(
    dtype, temperature
):
    query = torch.eye(2, device=TEST_DEVICE, dtype=dtype, requires_grad=True)
    model = losses.SoftTripleLoss(
        2,
        2,
        centers_per_class=2,
        gamma=temperature,
        la=2,
        margin=0.125,
        distance=distances.DotProductSimilarity(normalize_embeddings=False),
    ).to(device=TEST_DEVICE, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(
            model.weight.new_tensor(
                [[0.75, 0.25], [0.5, -0.5], [0.25, 0.75], [-0.5, 0.5]]
            )
        )
    query_ref = query.detach().double().requires_grad_()
    weight_ref = model.weight.detach().double().requires_grad_()
    reference_classes = []
    for centers in weight_ref.split(2):
        scores = query_ref @ centers.T
        probability = torch.softmax(scores / temperature, dim=1)
        reference_classes.append((probability * scores).sum(dim=1))
    expected_logits = torch.stack(reference_classes, dim=1)
    expected = F.cross_entropy(
        2 * (expected_logits - torch.eye(2, device=TEST_DEVICE) * 0.125),
        torch.arange(2, device=TEST_DEVICE),
    )
    actual = model(query, torch.arange(2, device=TEST_DEVICE))
    actual_gradients = torch.autograd.grad(actual, (query, model.weight))
    expected_gradients = torch.autograd.grad(expected, (query_ref, weight_ref))
    tolerance = (
        2e-2 if dtype == torch.bfloat16 else 3e-3 if dtype == torch.float16 else 3e-6
    )
    torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=1e-8)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient.double(), expected_gradient, rtol=tolerance, atol=2e-6
        )


def test_soft_triple_centers_before_float32_temperature_product_overflows():
    model = losses.SoftTripleLoss(
        2,
        1,
        centers_per_class=2,
        gamma=1e-10,
        distance=distances.DotProductSimilarity(normalize_embeddings=False),
    ).to(TEST_DEVICE)
    with torch.no_grad():
        model.weight.fill_(1e30)
    query = torch.ones(1, 1, device=TEST_DEVICE, requires_grad=True)
    logits = model.get_logits(query)
    torch.testing.assert_close(logits, torch.full_like(logits, 1e30))
    gradient = torch.autograd.grad(logits.sum(), query)[0]
    torch.testing.assert_close(gradient, torch.full_like(query, 2e30))


def test_soft_triple_empty_logits_keep_class_shape():
    model = losses.SoftTripleLoss(2, 3).to(TEST_DEVICE)
    logits = model.get_logits(torch.empty(0, 3, device=TEST_DEVICE))
    assert logits.shape == (0, 2)
