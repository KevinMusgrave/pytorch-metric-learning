import pytest
import torch
from torch.nn import functional as F

from pytorch_metric_learning import losses

from .. import TEST_DEVICE


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("tilt", [0.02, 0.04, 0.2])
@pytest.mark.parametrize("classes", [1, 2])
def test_p2s_near_center_preserves_parameter_and_embedding_gradients(
    dtype, tilt, classes
):
    model = losses.P2SGradLoss(classes, 2).to(TEST_DEVICE, dtype)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2, device=TEST_DEVICE, dtype=dtype)[:classes])
    query = torch.tensor(
        [[1.0, tilt]], device=TEST_DEVICE, dtype=dtype, requires_grad=True
    )
    labels = torch.tensor([0], device=TEST_DEVICE)
    value = model(query, labels)
    gradients = torch.autograd.grad(value, (query, model.weight), retain_graph=True)
    reference = query.detach().double().requires_grad_()
    weights = model.weight.detach().double().requires_grad_()
    normalized = reference / torch.linalg.vector_norm(reference, dim=1, keepdim=True)
    scores = (normalized @ weights.t()).clamp(-1, 1)
    expected = (scores - F.one_hot(labels, classes)).square().mean()
    expected_gradients = torch.autograd.grad(expected, (reference, weights))
    tolerance = 8e-3 if dtype in (torch.float16, torch.bfloat16) else 2e-3
    torch.testing.assert_close(value, expected.to(dtype), rtol=tolerance, atol=0)
    for actual, correct in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, correct.to(dtype), rtol=tolerance, atol=1e-9)
    assert model.weight.dtype == dtype
    assert value.dtype == dtype
    if classes == 1:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        value.backward()
        optimizer.step()
        assert model.weight[0, 1] > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_p2s_autocast_retains_near_center_learning_signal(dtype):
    model = losses.P2SGradLoss(1, 2).to(TEST_DEVICE, dtype)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0]], device=TEST_DEVICE, dtype=dtype))
    query = torch.tensor(
        [[1.0, 0.02]], device=TEST_DEVICE, dtype=dtype, requires_grad=True
    )
    with torch.autocast(TEST_DEVICE.type, dtype=torch.bfloat16):
        actual = model(query, torch.tensor([0], device=TEST_DEVICE))
    gradient = torch.autograd.grad(actual, model.weight)[0]
    assert actual.dtype == torch.float32
    assert 3.9e-8 < actual < 4.1e-8
    assert gradient[0, 0] < -0.00039
    assert gradient[0, 1] < -7.8e-6


def test_p2s_half_precision_tiny_rows_retain_original_epsilon():
    model = losses.P2SGradLoss(1, 2).to(TEST_DEVICE, torch.float16)
    with torch.no_grad():
        model.weight.copy_(
            torch.tensor([[1.0, 0.0]], device=TEST_DEVICE, dtype=torch.float16)
        )
    query = torch.tensor(
        [[1e-7, 0.0]], device=TEST_DEVICE, dtype=torch.float16, requires_grad=True
    )
    actual = model(query, torch.tensor([0], device=TEST_DEVICE))
    reference = query.detach().double().requires_grad_()
    expected = (reference[0, 0] / (2.0**-14) - 1).square()
    gradient = torch.autograd.grad(actual, query)[0]
    expected_gradient = torch.autograd.grad(expected, reference)[0]
    torch.testing.assert_close(actual, expected.half())
    torch.testing.assert_close(gradient, expected_gradient.half())
