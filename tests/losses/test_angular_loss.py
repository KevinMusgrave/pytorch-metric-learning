import unittest

import numpy as np
import torch

from pytorch_metric_learning.losses import AngularLoss
from pytorch_metric_learning.utils import common_functions as c_f

from .. import TEST_DEVICE, TEST_DTYPES


class TestAngularLoss(unittest.TestCase):
    def test_angular_loss_without_ref(self):
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(
                5, 32, requires_grad=True, device=TEST_DEVICE, dtype=dtype
            )
            embeddings = torch.nn.functional.normalize(embeddings)
            labels = torch.LongTensor([0, 0, 1, 1, 2])

            triplets = [
                (0, 1, 2),
                (0, 1, 3),
                (0, 1, 4),
                (1, 0, 2),
                (1, 0, 3),
                (1, 0, 4),
                (2, 3, 0),
                (2, 3, 1),
                (2, 3, 4),
                (3, 2, 0),
                (3, 2, 1),
                (3, 2, 4),
            ]

            self.helper(embeddings, labels, triplets, dtype)

    def test_angular_loss_with_ref(self):
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(
                4, 32, requires_grad=True, device=TEST_DEVICE, dtype=dtype
            )
            embeddings = torch.nn.functional.normalize(embeddings)
            labels = torch.LongTensor([0, 0, 1, 1])

            ref_emb = torch.randn(
                3, 32, requires_grad=True, device=TEST_DEVICE, dtype=dtype
            )
            ref_emb = torch.nn.functional.normalize(ref_emb)
            ref_labels = torch.LongTensor([0, 1, 2])

            triplets = [
                (0, 0, 1),
                (0, 0, 2),
                (1, 0, 1),
                (1, 0, 2),
                (2, 1, 0),
                (2, 1, 2),
                (3, 1, 0),
                (3, 1, 2),
            ]

            self.helper(embeddings, labels, triplets, dtype, ref_emb, ref_labels)

    def helper(
        self, embeddings, labels, triplets, dtype, ref_emb=None, ref_labels=None
    ):
        loss_func = AngularLoss(alpha=40)
        loss = loss_func(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
        loss.backward()
        sq_tan_alpha = (
            torch.tan(torch.tensor(np.radians(40), dtype=dtype).to(TEST_DEVICE)) ** 2
        )
        correct_losses = [0, 0, 0, 0]
        for a, p, n in triplets:
            anchor = embeddings[a]
            if ref_emb is not None:
                positive, negative = ref_emb[p], ref_emb[n]
            else:
                positive, negative = embeddings[p], embeddings[n]
            exponent = 4 * sq_tan_alpha * torch.matmul(
                anchor + positive, negative
            ) - 2 * (1 + sq_tan_alpha) * torch.matmul(anchor, positive)
            correct_losses[a] += torch.exp(exponent)
        total_loss = 0
        for c in correct_losses:
            total_loss += torch.log(1 + c)
        total_loss /= len(correct_losses)
        self.assertTrue(torch.isclose(loss, total_loss, rtol=1e-2))

    def test_with_no_valid_triplets(self):
        loss_func = AngularLoss(alpha=40)
        for dtype in TEST_DTYPES:
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor(
                [c_f.angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.LongTensor([0, 1, 2, 3, 4])
            loss = loss_func(embeddings, labels)
            loss.backward()
            self.assertEqual(loss, 0)
