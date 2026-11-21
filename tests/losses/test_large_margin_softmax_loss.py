import math
import unittest
from functools import partial
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pytorch_metric_learning.losses import LargeMarginSoftmaxLoss, SphereFaceLoss

from .. import TEST_DEVICE, TEST_DTYPES, helpers
from ..helpers import angle_to_coord

UNNORMALIZED_EMBEDDING_SCALE = 10


def index_of_angle_interval(angle, margin):
    return (angle / (math.pi / margin)).floor()


class TestLargeMarginSoftmaxLoss(unittest.TestCase):
    def test_large_margin_softmax_and_sphereface_loss(self):
        margin = 10
        scale = 2
        for dtype in TEST_DTYPES:
            loss_funcA = LargeMarginSoftmaxLoss(
                margin=margin, scale=scale, num_classes=10, embedding_size=2
            ).to(TEST_DEVICE, dtype)
            loss_funcB = SphereFaceLoss(
                margin=margin, scale=scale, num_classes=10, embedding_size=2
            ).to(TEST_DEVICE, dtype)

            embedding_angles = torch.arange(0, 180)
            embeddings = torch.tensor(
                np.array([angle_to_coord(a) for a in embedding_angles])
                * UNNORMALIZED_EMBEDDING_SCALE,
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.randint(low=0, high=10, size=(180,)).to(TEST_DEVICE)

            lossA = loss_funcA(embeddings, labels)
            lossB = loss_funcB(embeddings, labels)

            exact = embeddings.detach().double()
            weightsA = loss_funcA.weight.detach().double().t()
            weightsB = F.normalize(loss_funcB.weight.detach().double(), dim=1).t()

            product_of_magnitudesA = torch.norm(weightsA, p=2, dim=0).unsqueeze(
                0
            ) * torch.norm(exact, p=2, dim=1).unsqueeze(1)
            product_of_magnitudesB = torch.norm(weightsB, p=2, dim=0).unsqueeze(
                0
            ) * torch.norm(exact, p=2, dim=1).unsqueeze(1)
            cosinesA = torch.matmul(exact, weightsA) / (product_of_magnitudesA)
            cosinesB = torch.matmul(exact, weightsB) / (product_of_magnitudesB)
            coefficients = [math.comb(margin, 2 * n) for n in range((margin // 2) + 1)]

            for i, j in enumerate(labels):
                curr_cosineA = cosinesA[i, j]
                curr_cosineB = cosinesB[i, j]
                cos_with_marginA = torch.zeros(len(coefficients), dtype=torch.float64)
                cos_with_marginB = torch.zeros(len(coefficients), dtype=torch.float64)
                for z, c in enumerate(coefficients):
                    curr_valA = (
                        c
                        * (curr_cosineA ** (margin - (2 * z)))
                        * ((1 - curr_cosineA**2) ** z)
                    )
                    curr_valB = (
                        c
                        * (curr_cosineB ** (margin - (2 * z)))
                        * ((1 - curr_cosineB**2) ** z)
                    )
                    if z % 2 == 1:
                        curr_valA *= -1
                        curr_valB *= -1
                    cos_with_marginA[z] = curr_valA
                    cos_with_marginB[z] = curr_valB

                cos_with_marginA = torch.sum(cos_with_marginA)
                cos_with_marginB = torch.sum(cos_with_marginB)
                angleA = torch.acos(torch.clamp(curr_cosineA, -1 + 1e-7, 1 - 1e-7))
                angleB = torch.acos(torch.clamp(curr_cosineB, -1 + 1e-7, 1 - 1e-7))
                kA = index_of_angle_interval(angleA, margin)
                kB = index_of_angle_interval(angleB, margin)
                cosinesA[i, j] = ((-1) ** kA) * cos_with_marginA - (2 * kA)
                cosinesB[i, j] = ((-1) ** kB) * cos_with_marginB - (2 * kB)

            cosinesA *= product_of_magnitudesA
            cosinesB *= product_of_magnitudesB

            correct_lossA = torch.nn.functional.cross_entropy(cosinesA * scale, labels)
            correct_lossB = torch.nn.functional.cross_entropy(cosinesB * scale, labels)

            helpers.assert_close(lossA, correct_lossA, dtype, atol=1e-8)
            helpers.assert_close(lossB, correct_lossB, dtype, atol=1e-8)

            logits_outA = loss_funcA.get_logits(embeddings)
            logits_outB = loss_funcB.get_logits(embeddings)
            helpers.assert_close(
                logits_outA / (scale * product_of_magnitudesA),
                exact @ weightsA / product_of_magnitudesA,
                dtype,
            )
            helpers.assert_close(
                logits_outB / (scale * product_of_magnitudesB),
                exact @ weightsB / product_of_magnitudesB,
                dtype,
            )

    def test_backward(self):
        generator = torch.Generator().manual_seed(5)
        embeddings = 3 * torch.randn(24, 2, generator=generator)
        weight = torch.randn(5, 2, generator=generator)
        labels = torch.randint(0, 5, (24,), generator=generator).to(TEST_DEVICE)
        for margin in (2, 4):
            for loss_class, scale_by_weight_norm in (
                (LargeMarginSoftmaxLoss, True),
                (SphereFaceLoss, False),
            ):
                for dtype in TEST_DTYPES:
                    loss_func = loss_class(
                        num_classes=5, embedding_size=2, margin=margin, scale=1.5
                    ).to(TEST_DEVICE, dtype)
                    with self.subTest(loss=loss_class.__name__, m=margin, dtype=dtype):
                        helpers.assert_matches_float64(
                            partial(call_with_weight, loss_func, labels=labels),
                            partial(
                                angular_margin_reference,
                                labels=labels,
                                margin=margin,
                                scale=1.5,
                                weight_norms=scale_by_weight_norm,
                            ),
                            [embeddings, weight],
                            dtype,
                        )

    def test_margin_must_be_a_positive_integer(self):
        fractional: Any = 2.5
        with self.assertRaises(TypeError):
            LargeMarginSoftmaxLoss(3, 4, margin=fractional)
        with self.assertRaises(ValueError):
            LargeMarginSoftmaxLoss(3, 4, margin=0)


def call_with_weight(loss_func, embeddings, weight, labels):
    return torch.func.functional_call(
        loss_func, {"weight": weight}, (embeddings, labels)
    )


def angular_margin_reference(embeddings, weight, labels, margin, scale, weight_norms):
    cosine = F.normalize(embeddings, dim=1) @ F.normalize(weight, dim=1).T
    theta = torch.acos(cosine.gather(1, labels.unsqueeze(1)).squeeze(1))
    k = torch.floor(theta.detach() * margin / math.pi)
    psi = (-1) ** k * torch.cos(margin * theta) - 2 * k
    target = F.one_hot(labels, len(weight)).bool()
    modified = torch.where(target, psi.unsqueeze(1), cosine)
    magnitudes = embeddings.norm(dim=1, keepdim=True)
    if weight_norms:
        magnitudes = magnitudes * weight.norm(dim=1)
    return F.cross_entropy(scale * magnitudes * modified, labels)


SCALED_NORM_CASES = {
    "norm_overflow": ([60000.0, 60000.0], [[0.5, 0.0], [0.0, 0.5]], 1, 1),
    "product_overflow": (
        [40000.0, 40000.0],
        [[-0.517578125, 1.931640625], [1.931640625, -0.517578125]],
        1,
        0.5,
    ),
    "angular_margin": (
        [40000.0, 40000.0],
        [[-0.517578125, 1.931640625], [1.931640625, -0.517578125]],
        4,
        1 / 64,
    ),
    "product_underflow": ([1e-4, 1e-4], [[1e-4, 0.0], [0.0, 1e-4]], 1, 1024),
}


def scaled_norm_fixture(case, dtype, loss_class):
    row, weights, margin, scale = SCALED_NORM_CASES[case]
    x = torch.tensor([row], dtype=dtype, device=TEST_DEVICE, requires_grad=True)
    loss = loss_class(2, 2, margin=margin, scale=scale).to(
        device=TEST_DEVICE, dtype=dtype
    )
    with torch.no_grad():
        loss.weight.copy_(torch.tensor(weights, dtype=dtype, device=TEST_DEVICE))
    return x, loss


@pytest.mark.parametrize("case", SCALED_NORM_CASES)
@pytest.mark.parametrize("loss_class", [LargeMarginSoftmaxLoss, SphereFaceLoss])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("label", [0, 1])
def test_finite_scaled_logits_and_training_gradients(case, loss_class, dtype, label):
    x, loss = scaled_norm_fixture(case, dtype, loss_class)
    labels = torch.tensor([label], device=TEST_DEVICE)
    exact_x = x.detach().double().requires_grad_()
    exact_weight = loss.weight.detach().double().requires_grad_()
    expected_weight = (
        F.normalize(exact_weight, dim=1)
        if loss_class is SphereFaceLoss
        else exact_weight
    )
    expected_logits = exact_x @ expected_weight.T * loss.scale
    logits = loss.get_logits(x)
    assert logits.dtype == dtype
    assert logits.isfinite().all()
    torch.testing.assert_close(
        logits.double(), expected_logits, rtol=helpers.rtol(dtype), atol=1e-8
    )
    actual = loss(x, labels)
    expected = (
        F.cross_entropy(expected_logits, labels)
        if loss.margin == 1
        else angular_margin_reference(
            exact_x,
            exact_weight,
            labels,
            loss.margin,
            loss.scale,
            loss_class is LargeMarginSoftmaxLoss,
        )
    )
    assert actual.dtype == dtype and actual.isfinite()
    helpers.assert_close(actual, expected, dtype)
    actual.backward()
    expected.backward()
    for actual_leaf, exact_leaf in [(x, exact_x), (loss.weight, exact_weight)]:
        assert actual_leaf.grad is not None and exact_leaf.grad is not None
        assert actual_leaf.grad.isfinite().all()
        helpers.assert_close(actual_leaf.grad, exact_leaf.grad, dtype)


@pytest.mark.parametrize(
    ("input_dtype", "autocast_dtype", "result_dtype"),
    [
        (torch.float32, torch.float16, torch.float32),
        (torch.float32, torch.bfloat16, torch.float32),
        (torch.float16, torch.float16, torch.float16),
        (torch.float16, torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.float16, torch.float32),
    ],
)
@pytest.mark.parametrize("loss_class", [LargeMarginSoftmaxLoss, SphereFaceLoss])
def test_norm_scaling_keeps_autocast_result_promotion(
    input_dtype, autocast_dtype, result_dtype, loss_class
):
    x, loss = scaled_norm_fixture("norm_overflow", input_dtype, loss_class)
    labels = torch.tensor([0], device=TEST_DEVICE)
    exact_x = x.detach().double().requires_grad_()
    exact_weight = loss.weight.detach().double().requires_grad_()
    expected_weight = (
        F.normalize(exact_weight, dim=1)
        if loss_class is SphereFaceLoss
        else exact_weight
    )
    expected = F.cross_entropy(exact_x @ expected_weight.T, labels)
    with torch.autocast(TEST_DEVICE.type, dtype=autocast_dtype):
        logits = loss.get_logits(x)
        actual = loss(x, labels)
    assert logits.dtype == result_dtype and logits.isfinite().all()
    assert actual.dtype == torch.float32 and actual.isfinite()
    helpers.assert_close(actual, expected, autocast_dtype)
    actual.backward()
    expected.backward()
    for actual_leaf, exact_leaf in [(x, exact_x), (loss.weight, exact_weight)]:
        assert actual_leaf.grad is not None and exact_leaf.grad is not None
        assert actual_leaf.grad.isfinite().all()
        helpers.assert_close(actual_leaf.grad, exact_leaf.grad, autocast_dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("size", [0, 1])
@pytest.mark.parametrize("loss_class", [LargeMarginSoftmaxLoss, SphereFaceLoss])
def test_norm_scaling_keeps_empty_and_zero_inputs(dtype, size, loss_class):
    loss = loss_class(2, 2, margin=1).to(TEST_DEVICE, dtype)
    x = torch.zeros(size, 2, dtype=dtype, device=TEST_DEVICE, requires_grad=True)
    logits = loss.get_logits(x)
    assert logits.shape == (size, 2) and logits.dtype == dtype
    assert torch.count_nonzero(logits) == 0
    actual = loss(x, torch.zeros(size, dtype=torch.long, device=TEST_DEVICE))
    helpers.assert_close(actual, math.log(2) if size else 0, dtype)
    actual.backward()
    assert x.grad is not None and torch.count_nonzero(x.grad) == 0


@pytest.mark.parametrize("loss_class", [LargeMarginSoftmaxLoss, SphereFaceLoss])
def test_final_logits_keep_float16_range(loss_class):
    value = 60000.0 if loss_class is SphereFaceLoss else 40000.0
    x = torch.full((1, 2), value, dtype=torch.float16, device=TEST_DEVICE)
    loss = loss_class(2, 2, margin=1).to(TEST_DEVICE, torch.float16)
    with torch.no_grad():
        weight = (
            torch.ones(2, 2, device=TEST_DEVICE, dtype=torch.float16)
            if loss_class is SphereFaceLoss
            else 2 * torch.eye(2, device=TEST_DEVICE, dtype=torch.float16)
        )
        loss.weight.copy_(weight)
    logits = loss.get_logits(x)
    assert logits.dtype == torch.float16
    assert logits.isinf().all()
