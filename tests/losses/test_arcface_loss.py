import unittest

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pytorch_metric_learning.losses import (
    ArcFaceLoss,
    DynamicArcFaceLoss,
    SubCenterArcFaceLoss,
)

from .. import TEST_DEVICE, TEST_DTYPES, helpers
from ..helpers import angle_to_coord


class TestArcFaceLoss(unittest.TestCase):
    def test_arcface_loss(self):
        margin = np.radians(30)
        scale = 64
        for dtype in TEST_DTYPES:
            loss_func = ArcFaceLoss(
                margin=margin, scale=scale, num_classes=10, embedding_size=2
            ).to(TEST_DEVICE, dtype)
            embedding_angles = torch.arange(0, 180)
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.randint(low=0, high=10, size=(180,)).to(TEST_DEVICE)

            loss = loss_func(embeddings, labels)
            loss.backward()

            weights = F.normalize(loss_func.weight.detach().double(), p=2, dim=1).t()
            cosines = torch.matmul(F.normalize(embeddings.detach().double()), weights)
            logits = cosines.clone()

            for i, c in enumerate(labels):
                acos = torch.acos(torch.clamp(logits[i, c], -1, 1))
                if acos <= (np.pi - margin):
                    logits[i, c] = torch.cos(acos + margin)
                else:
                    logits[i, c] -= margin * np.sin(margin)

            correct_loss = F.cross_entropy(logits * scale, labels)
            helpers.assert_close(loss, correct_loss, dtype, atol=1e-8)

            logits_out = loss_func.get_logits(embeddings)
            helpers.assert_close(logits_out / scale, cosines, dtype)


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("margin", [0.0, 0.5])
def test_arcface_endpoint_values_and_gradients(dtype, margin):
    cosine = torch.tensor([-1.0, -0.3, 0.2, 1.0], device=TEST_DEVICE, dtype=dtype)
    cosine.requires_grad_()
    labels = torch.zeros(4, dtype=torch.long, device=TEST_DEVICE)
    loss = ArcFaceLoss(2, 2, margin=margin).to(TEST_DEVICE, dtype)
    actual = loss.modify_cosine_of_target_classes(cosine, labels)
    actual.sum().backward()
    expected = cosine.detach().double()
    if margin:
        angles = expected.acos()
        expected = torch.where(
            angles <= np.pi - margin,
            (angles + margin).cos(),
            expected - margin * np.sin(margin),
        )
    helpers.assert_close(actual, expected, dtype)
    expected_gradient = torch.ones_like(cosine, dtype=torch.float64)
    if margin:
        interior = cosine.detach().double()[1:-1]
        expected_gradient[1:-1] = np.cos(margin) + (
            interior * np.sin(margin) / torch.sqrt(1 - interior.square())
        )
        expected_gradient[-1] = 0
    assert cosine.grad is not None
    helpers.assert_close(cosine.grad, expected_gradient, dtype)


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("variant", ["arcface", "subcenter", "dynamic"])
def test_arcface_endpoint_training_step_stays_finite(dtype, variant):
    if variant == "dynamic":
        loss = DynamicArcFaceLoss([2, 5], 2, sub_centers=1)
    elif variant == "subcenter":
        loss = SubCenterArcFaceLoss(2, 2, sub_centers=1)
    else:
        loss = ArcFaceLoss(2, 2)
    loss = loss.to(TEST_DEVICE, dtype)
    with torch.no_grad():
        loss.weight.copy_(torch.eye(2, device=TEST_DEVICE, dtype=dtype))
    embeddings = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
        device=TEST_DEVICE,
        dtype=dtype,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1], device=TEST_DEVICE)
    optimizer = torch.optim.SGD(loss.parameters(), lr=0.01)
    value = loss(embeddings, labels)
    value.backward()
    assert torch.isfinite(value)
    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()
    assert loss.weight.grad is not None and torch.isfinite(loss.weight.grad).all()
    optimizer.step()
    assert torch.isfinite(loss.weight).all()


def test_arcface_zero_margin_keeps_near_endpoint_derivative():
    cosine = torch.tensor(
        [-1.0, -1.0 + 1e-10, 1.0 - 1e-10, 1.0],
        dtype=torch.float64,
        device=TEST_DEVICE,
        requires_grad=True,
    )
    labels = torch.zeros(4, dtype=torch.long, device=TEST_DEVICE)
    loss = ArcFaceLoss(2, 2, margin=0).to(TEST_DEVICE, torch.float64)
    result = loss.modify_cosine_of_target_classes(cosine, labels)
    result.sum().backward()
    torch.testing.assert_close(result, cosine, rtol=0, atol=0)
    torch.testing.assert_close(cosine.grad, torch.ones_like(cosine), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("margin", [0.0, 0.5])
@pytest.mark.parametrize("variant", ["arcface", "subcenter", "dynamic"])
def test_arcface_rounded_endpoint_keeps_training_gradients(dtype, margin, variant):
    if variant == "dynamic":
        loss = DynamicArcFaceLoss(
            [2, 5], 2, sub_centers=1, margin_bounds=(margin, margin + 0.5), scale=1
        )
    elif variant == "subcenter":
        loss = SubCenterArcFaceLoss(2, 2, margin=margin, sub_centers=1, scale=1)
    else:
        loss = ArcFaceLoss(2, 2, margin=margin, scale=1)
    loss = loss.to(TEST_DEVICE, dtype)
    with torch.no_grad():
        loss.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]], device=TEST_DEVICE))
    offset = {torch.float16: 0.01, torch.float32: 1e-5, torch.float64: 1e-8}[dtype]
    x = torch.tensor(
        [[1.0, offset]], dtype=dtype, device=TEST_DEVICE, requires_grad=True
    )
    labels = torch.tensor([1], device=TEST_DEVICE)
    actual = loss(x, labels)
    actual.backward()
    reference_x = x.detach().double().requires_grad_()
    reference_weight = loss.weight.detach().double().requires_grad_()
    cosine = F.normalize(reference_x, dim=1) @ F.normalize(reference_weight, dim=1).T
    logits = cosine.clone()
    logits[:, 1] = cosine[:, 1] - margin * np.sin(margin)
    expected = F.cross_entropy(logits, labels)
    expected.backward()
    helpers.assert_close(actual, expected, dtype)
    assert x.grad is not None and reference_x.grad is not None
    assert loss.weight.grad is not None and reference_weight.grad is not None
    tolerance = 3e-3 if dtype == torch.float16 else 1e-6
    torch.testing.assert_close(
        x.grad[:, 1].double(), reference_x.grad[:, 1], rtol=tolerance, atol=0
    )
    torch.testing.assert_close(
        loss.weight.grad.double(), reference_weight.grad, rtol=tolerance, atol=1e-12
    )
