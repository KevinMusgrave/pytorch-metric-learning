import math

import pytest
import torch

from pytorch_metric_learning.losses import GE2ELoss

from .. import TEST_DEVICE
from .test_ge2e_loss import equations_6_to_9


@pytest.mark.parametrize(
    "dtype,offset",
    [
        (torch.float16, 1000.0),
        (torch.bfloat16, 1e8),
        (torch.float32, 1e8),
        (torch.float64, 1e16),
    ],
)
@pytest.mark.parametrize("sign", [-1, 0, 1])
def test_softmax_bias_preserves_the_objective_and_gradients(dtype, offset, sign):
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    embeddings = torch.tensor(
        vectors, dtype=dtype, requires_grad=True, device=TEST_DEVICE
    )
    labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
    loss = GE2ELoss(init_w=2, init_b=sign * offset).to(TEST_DEVICE).to(dtype)
    reference = embeddings.detach().double().requires_grad_()
    scale = loss.w.detach().double().requires_grad_()
    expected = equations_6_to_9(reference, labels.tolist(), scale, 0.0, "softmax")
    expected_input_gradient, expected_scale_gradient = torch.autograd.grad(
        expected, (reference, scale)
    )
    actual = loss(embeddings, labels)
    actual.backward()
    assert (
        embeddings.grad is not None
        and loss.raw_w.grad is not None
        and loss.b.grad is not None
    )
    rtol = 0.02 if dtype in (torch.float16, torch.bfloat16) else 1e-6
    torch.testing.assert_close(actual.double(), expected, rtol=rtol, atol=1e-7)
    torch.testing.assert_close(
        embeddings.grad.double(), expected_input_gradient, rtol=rtol, atol=1e-7
    )
    expected_raw_gradient = expected_scale_gradient * torch.sigmoid(
        loss.raw_w.detach().double()
    )
    torch.testing.assert_close(
        loss.raw_w.grad.double(), expected_raw_gradient, rtol=rtol, atol=1e-7
    )
    torch.testing.assert_close(loss.b.grad, torch.zeros_like(loss.b), rtol=0, atol=0)


@pytest.mark.parametrize("dtype,scale", [(torch.float32, 1e8), (torch.float64, 1e16)])
@pytest.mark.parametrize("classes", [2, 3])
def test_large_equal_scores_retain_log_class_count(dtype, scale, classes):
    embeddings = torch.tensor(
        [[1.0, 0.0]] * (2 * classes),
        dtype=dtype,
        requires_grad=True,
        device=TEST_DEVICE,
    )
    labels = torch.arange(classes, device=TEST_DEVICE).repeat_interleave(2)
    loss = GE2ELoss(init_w=scale, init_b=0).to(TEST_DEVICE).to(dtype)
    actual = loss(embeddings, labels)
    torch.testing.assert_close(
        actual, torch.tensor(math.log(classes), dtype=dtype, device=TEST_DEVICE)
    )
    actual.backward()
    assert (
        embeddings.grad is not None
        and loss.raw_w.grad is not None
        and loss.b.grad is not None
    )
    assert torch.isfinite(embeddings.grad).all()
    torch.testing.assert_close(
        loss.raw_w.grad, torch.zeros_like(loss.raw_w), rtol=0, atol=1e-7
    )
    torch.testing.assert_close(loss.b.grad, torch.zeros_like(loss.b), rtol=0, atol=0)


@pytest.mark.parametrize("bias", [-2.5, 2.5])
def test_contrast_variant_keeps_its_bias_dependence(bias):
    embeddings = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
        device=TEST_DEVICE,
    )
    labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
    loss = GE2ELoss(variant="contrast", init_w=2, init_b=bias).to(TEST_DEVICE).double()
    actual = loss(embeddings, labels)
    reference_bias = torch.tensor(
        bias, dtype=torch.float64, requires_grad=True, device=TEST_DEVICE
    )
    expected = equations_6_to_9(
        embeddings, labels.tolist(), loss.w, reference_bias, "contrast"
    )
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual, loss.b)[0]
    expected_gradient = torch.autograd.grad(expected, reference_bias)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    assert actual_gradient.abs() > 0
