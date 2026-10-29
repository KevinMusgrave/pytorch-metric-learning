import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_metric_learning.losses import AngularLoss

from .. import TEST_DEVICE, TEST_DTYPES
from ..helpers import angle_to_coord
from .utils import get_triplet_embeddings_with_ref


class TestAngularLoss(unittest.TestCase):
    def test_angular_loss_without_ref(self):
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(
                5, 32, requires_grad=True, device=TEST_DEVICE, dtype=dtype
            )
            embeddings = torch.nn.functional.normalize(embeddings)
            labels = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)

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
            (
                embeddings,
                labels,
                ref_emb,
                ref_labels,
                triplets,
            ) = get_triplet_embeddings_with_ref(dtype, TEST_DEVICE)

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
        correct_losses = torch.zeros(4, dtype=dtype, device=TEST_DEVICE)
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
        total_loss = torch.log(1 + correct_losses).mean()
        self.assertTrue(torch.isclose(loss, total_loss, rtol=1e-2))

    def test_with_no_valid_triplets(self):
        loss_func = AngularLoss(alpha=40)
        for dtype in TEST_DTYPES:
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
            loss = loss_func(embeddings, labels)
            loss.backward()
            self.assertEqual(loss, 0)


class TestAngularLossNormalization(unittest.TestCase):
    def test_negatives_are_normalized_like_anchors_and_positives(self):
        torch.manual_seed(5)
        embeddings = 3 * torch.randn(6, 4, dtype=torch.float64)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        alpha = 40
        loss = AngularLoss(alpha=alpha)(embeddings, labels)
        x = F.normalize(embeddings, dim=1)
        tan_sq = math.tan(math.radians(alpha)) ** 2
        terms = []
        for a in range(6):
            for p in range(6):
                if a == p or labels[a] != labels[p]:
                    continue
                f = [
                    4 * tan_sq * (x[a] + x[p]) @ x[n] - 2 * (1 + tan_sq) * x[a] @ x[p]
                    for n in range(6)
                    if labels[n] != labels[a]
                ]
                terms.append(torch.log1p(torch.exp(torch.stack(f)).sum()))
        torch.testing.assert_close(loss, torch.stack(terms).mean())
        torch.testing.assert_close(AngularLoss(alpha=alpha)(x, labels), loss)
