import math

import pytest
import torch
import torch.nn.functional as F

from pytorch_metric_learning import distances, losses

from .. import TEST_DEVICE


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("temperature", [1e-5, 0.05, -1e-5])
def test_normalized_softmax_preserves_representable_equal_class_gradients(
    dtype, temperature
):
    model = losses.NormalizedSoftmaxLoss(2, 2, temperature=temperature).to(
        device=TEST_DEVICE, dtype=dtype
    )
    with torch.no_grad():
        model.weight.copy_(model.weight.new_tensor([[1, 0], [1, 0]]))
    query = torch.tensor(
        [[1.0, 1.0]], device=TEST_DEVICE, dtype=dtype, requires_grad=True
    )
    logits = model.get_logits(query)
    output_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    assert logits.dtype == output_dtype
    tolerance = (
        2e-2 if dtype == torch.bfloat16 else 3e-3 if dtype == torch.float16 else 3e-6
    )
    torch.testing.assert_close(
        logits,
        logits.new_full((1, 2), 1 / (math.sqrt(2) * temperature)),
        rtol=tolerance,
        atol=0,
    )
    value = model(query, torch.tensor([0], device=TEST_DEVICE))
    torch.testing.assert_close(
        value, value.new_tensor(math.log(2)), rtol=tolerance, atol=0
    )
    value.backward()
    torch.testing.assert_close(query.grad, torch.zeros_like(query), rtol=0, atol=1e-6)
    expected = model.weight.new_tensor(
        [
            [0, -1 / (2 * math.sqrt(2) * temperature)],
            [0, 1 / (2 * math.sqrt(2) * temperature)],
        ]
    )
    torch.testing.assert_close(model.weight.grad, expected, rtol=tolerance, atol=0)
    previous = model.weight.detach().clone()
    torch.optim.SGD(model.parameters(), lr=abs(temperature)).step()
    assert torch.isfinite(model.weight).all()
    assert not torch.equal(previous, model.weight)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("temperature", [1e-5, 0.05])
def test_normalized_softmax_public_logits_keep_absolute_scale(dtype, temperature):
    model = losses.NormalizedSoftmaxLoss(
        2,
        2,
        temperature=temperature,
        distance=distances.DotProductSimilarity(normalize_embeddings=False),
    ).to(device=TEST_DEVICE, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2, device=TEST_DEVICE, dtype=dtype))
    query = torch.tensor(
        [[2.0, 3.0], [-1.0, 4.0]], device=TEST_DEVICE, dtype=dtype, requires_grad=True
    )
    logits = model.get_logits(query)
    expected = query.detach().double() / temperature
    torch.testing.assert_close(logits.double(), expected, rtol=3e-6, atol=0)
    gradient = torch.autograd.grad((logits * temperature).sum(), query)[0]
    torch.testing.assert_close(gradient, torch.ones_like(query), rtol=2e-3, atol=0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_normalized_softmax_distance_polarity_and_ordinary_objective(dtype):
    model = losses.NormalizedSoftmaxLoss(
        2,
        2,
        temperature=0.5,
        distance=distances.LpDistance(normalize_embeddings=False, power=2),
    ).to(device=TEST_DEVICE, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2, device=TEST_DEVICE, dtype=dtype))
    query = torch.tensor(
        [[0.5, 0.25], [0.25, 0.5]], device=TEST_DEVICE, dtype=dtype, requires_grad=True
    )
    reference = query.detach().double().requires_grad_()
    weights = model.weight.detach().double().requires_grad_()
    expected_logits = (
        -(reference[:, None] - F.normalize(weights, dim=1)[None]).square().sum(dim=2)
        / 0.5
    )
    labels = torch.arange(2, device=TEST_DEVICE)
    expected = F.cross_entropy(expected_logits, labels)
    actual = model(query, labels)
    actual_gradients = torch.autograd.grad(actual, (query, model.weight))
    expected_gradients = torch.autograd.grad(expected, (reference, weights))
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


def test_normalized_softmax_empty_logits_keep_class_shape():
    model = losses.NormalizedSoftmaxLoss(2, 3).to(TEST_DEVICE)
    logits = model.get_logits(torch.empty(0, 3, device=TEST_DEVICE))
    assert logits.shape == (0, 2)


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_normalized_softmax_preserves_autocast_loss_precision(input_dtype):
    model = losses.NormalizedSoftmaxLoss(2, 2).to(TEST_DEVICE)
    query = torch.tensor(
        [[1.0, 0.25]], device=TEST_DEVICE, dtype=input_dtype, requires_grad=True
    )
    with torch.autocast(TEST_DEVICE.type, dtype=torch.bfloat16):
        value = model(query, torch.tensor([0], device=TEST_DEVICE))
    assert value.dtype == torch.float32
    value.backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()


def test_normalized_softmax_preserves_custom_distance_precision():
    class DoubleSimilarity(distances.DotProductSimilarity):
        def compute_mat(self, query_emb, ref_emb):
            return query_emb.double() @ ref_emb.double().T

    model = losses.NormalizedSoftmaxLoss(
        2, 2, distance=DoubleSimilarity(normalize_embeddings=True)
    ).to(TEST_DEVICE)
    query = torch.tensor([[1.0, 0.25]], device=TEST_DEVICE, requires_grad=True)
    value = model(query, torch.tensor([0], device=TEST_DEVICE))
    assert value.dtype == torch.float64
    value.backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
