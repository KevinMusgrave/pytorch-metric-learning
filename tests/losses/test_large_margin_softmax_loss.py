import math
import unittest

import numpy as np
import scipy
import torch

from pytorch_metric_learning.losses import LargeMarginSoftmaxLoss, SphereFaceLoss
from pytorch_metric_learning.utils import common_functions as c_f

from .. import TEST_DEVICE, TEST_DTYPES


class TestLargeMarginSoftmaxLoss(unittest.TestCase):
    def test_large_margin_softmax_and_sphereface_loss(self):
        margin = 10
        scale = 2
        for dtype in TEST_DTYPES:
            loss_funcA = LargeMarginSoftmaxLoss(
                margin=margin, scale=scale, num_classes=10, embedding_size=2
            )
            loss_funcB = SphereFaceLoss(
                margin=margin, scale=scale, num_classes=10, embedding_size=2
            )

            embedding_angles = torch.arange(0, 180)
            # multiply by 10 to make the embeddings unnormalized
            embeddings = torch.tensor(
                np.array([c_f.angle_to_coord(a) for a in embedding_angles]) * 10,
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.randint(low=0, high=10, size=(180,)).to(TEST_DEVICE)

            lossA = loss_funcA(embeddings, labels)
            lossB = loss_funcB(embeddings, labels)

            weightsA = loss_funcA.W
            weightsB = torch.nn.functional.normalize(loss_funcB.W, dim=0)

            product_of_magnitudesA = torch.norm(weightsA, p=2, dim=0).unsqueeze(
                0
            ) * torch.norm(embeddings, p=2, dim=1).unsqueeze(1)
            product_of_magnitudesB = torch.norm(weightsB, p=2, dim=0).unsqueeze(
                0
            ) * torch.norm(embeddings, p=2, dim=1).unsqueeze(1)
            cosinesA = torch.matmul(embeddings, weightsA) / (product_of_magnitudesA)
            cosinesB = torch.matmul(embeddings, weightsB) / (product_of_magnitudesB)
            coefficients = [
                scipy.special.binom(margin, 2 * n) for n in range((margin // 2) + 1)
            ]

            for i, j in enumerate(labels):
                curr_cosineA = cosinesA[i, j]
                curr_cosineB = cosinesB[i, j]
                cos_with_marginA = torch.zeros(len(coefficients))
                cos_with_marginB = torch.zeros(len(coefficients))
                for z, c in enumerate(coefficients):
                    curr_valA = (
                        c
                        * (curr_cosineA ** (margin - (2 * z)))
                        * ((1 - curr_cosineA ** 2) ** z)
                    )
                    curr_valB = (
                        c
                        * (curr_cosineB ** (margin - (2 * z)))
                        * ((1 - curr_cosineB ** 2) ** z)
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
                kA = (
                    angleA / (math.pi / margin)
                ).floor()  # Equation 6: angles needs to be between [k*pi/m and (k+1)*pi/m]
                kB = (
                    angleB / (math.pi / margin)
                ).floor()  # Equation 6: angles needs to be between [k*pi/m and (k+1)*pi/m]
                cosinesA[i, j] = ((-1) ** kA) * cos_with_marginA - (2 * kA)
                cosinesB[i, j] = ((-1) ** kB) * cos_with_marginB - (2 * kB)

            cosinesA *= product_of_magnitudesA
            cosinesB *= product_of_magnitudesB

            correct_lossA = torch.nn.functional.cross_entropy(cosinesA * scale, labels)
            correct_lossB = torch.nn.functional.cross_entropy(cosinesB * scale, labels)

            rtol = 1e-2 if dtype == torch.float16 else 1e-5
            self.assertTrue(torch.isclose(lossA, correct_lossA, rtol=rtol))
            self.assertTrue(torch.isclose(lossB, correct_lossB, rtol=rtol))

            # test get_logits
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
        margin = 10
        scale = 2
        for dtype in TEST_DTYPES:
            loss_funcA = LargeMarginSoftmaxLoss(
                margin=margin, scale=scale, num_classes=10, embedding_size=2
            )
            loss_funcB = SphereFaceLoss(
                margin=margin, scale=scale, num_classes=10, embedding_size=2
            )
            for loss_func in [loss_funcA, loss_funcB]:
                embedding_angles = torch.arange(0, 180)
                # multiply by 10 to make the embeddings unnormalized
                embeddings = torch.tensor(
                    np.array([c_f.angle_to_coord(a) for a in embedding_angles]) * 10,
                    requires_grad=True,
                    dtype=dtype,
                ).to(
                    TEST_DEVICE
                )  # 2D embeddings
                labels = torch.randint(low=0, high=10, size=(180,)).to(TEST_DEVICE)

                loss = loss_func(embeddings, labels)
                loss.backward()
