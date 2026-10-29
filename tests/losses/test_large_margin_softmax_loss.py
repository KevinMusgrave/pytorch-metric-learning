import math
import unittest
from functools import partial
from typing import Any

import numpy as np
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

            weightsA = loss_funcA.weight.t()
            weightsB = torch.nn.functional.normalize(loss_funcB.weight, dim=1).t()

            product_of_magnitudesA = torch.norm(weightsA, p=2, dim=0).unsqueeze(
                0
            ) * torch.norm(embeddings, p=2, dim=1).unsqueeze(1)
            product_of_magnitudesB = torch.norm(weightsB, p=2, dim=0).unsqueeze(
                0
            ) * torch.norm(embeddings, p=2, dim=1).unsqueeze(1)
            cosinesA = torch.matmul(embeddings, weightsA) / (product_of_magnitudesA)
            cosinesB = torch.matmul(embeddings, weightsB) / (product_of_magnitudesB)
            coefficients = [math.comb(margin, 2 * n) for n in range((margin // 2) + 1)]

            for i, j in enumerate(labels):
                curr_cosineA = cosinesA[i, j]
                curr_cosineB = cosinesB[i, j]
                cos_with_marginA = torch.zeros(len(coefficients))
                cos_with_marginB = torch.zeros(len(coefficients))
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

            rtol = helpers.rtol(dtype)
            self.assertTrue(torch.isclose(lossA, correct_lossA, rtol=rtol))
            self.assertTrue(torch.isclose(lossB, correct_lossB, rtol=rtol))

            if dtype != torch.float16:
                logits_outA = loss_funcA.get_logits(embeddings)
                logits_outB = loss_funcB.get_logits(embeddings)
                self.assertTrue(
                    torch.allclose(
                        logits_outA,
                        torch.matmul(embeddings, weightsA) * scale,
                        rtol=1e-2,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        logits_outB,
                        torch.matmul(embeddings, weightsB) * scale,
                        rtol=1e-2,
                    )
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
