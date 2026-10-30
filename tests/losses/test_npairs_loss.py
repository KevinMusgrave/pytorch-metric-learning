import unittest
from functools import partial

import torch

from pytorch_metric_learning.losses import NPairsLoss
from pytorch_metric_learning.regularizers import LpRegularizer

from .. import TEST_DEVICE, TEST_DTYPES, helpers
from ..helpers import angle_to_coord


class TestNPairsLoss(unittest.TestCase):
    def test_npairs_loss(self):
        loss_funcA = NPairsLoss()
        loss_funcB = NPairsLoss(embedding_regularizer=LpRegularizer(power=2))
        embedding_norm = 2.3

        for dtype in TEST_DTYPES:
            embedding_angles = list(range(0, 180, 20))[:7]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            embeddings_for_loss_fn = embeddings * embedding_norm
            labels = torch.tensor([0, 0, 1, 1, 1, 2, 3], dtype=torch.long).to(
                TEST_DEVICE
            )

            lossA = loss_funcA(embeddings_for_loss_fn, labels)
            lossB = loss_funcB(embeddings_for_loss_fn, labels)

            pos_pairs = [(0, 1), (2, 3)]
            neg_pairs = [(0, 3), (2, 1)]

            embeddings = torch.nn.functional.normalize(embeddings)
            total_loss = torch.zeros((), dtype=dtype, device=TEST_DEVICE)
            for a1, p in pos_pairs:
                anchor, positive = embeddings[a1], embeddings[p]
                numerator = torch.exp(torch.matmul(anchor, positive))
                denominator = numerator.clone()
                for a2, n in neg_pairs:
                    if a2 == a1:
                        negative = embeddings[n]
                        denominator += torch.exp(torch.matmul(anchor, negative))
                curr_loss = -torch.log(numerator / denominator)
                total_loss += curr_loss

            total_loss /= len(pos_pairs[0])
            self.assertTrue(torch.isclose(lossA, total_loss))
            self.assertTrue(
                torch.isclose(
                    lossB,
                    total_loss
                    + torch.mean(torch.norm(embeddings_for_loss_fn, dim=1) ** 2),
                )
            )

    def test_with_no_valid_pairs(self):
        for dtype in TEST_DTYPES:
            loss_func = NPairsLoss()
            embedding_angles = [0]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.tensor([0], dtype=torch.long).to(TEST_DEVICE)
            loss = loss_func(embeddings, labels)
            loss.backward()
            self.assertEqual(loss, 0)

    def test_backward(self):
        labels = torch.tensor([0, 0, 1, 1, 1, 2, 3], device=TEST_DEVICE)
        coordinates = 2.3 * torch.tensor([angle_to_coord(a) for a in range(0, 140, 20)])

        def reference(embeddings, with_regularizer):
            x = torch.nn.functional.normalize(embeddings, dim=1)
            similarities = x[[0, 2]] @ x[[1, 3]].T
            loss = torch.mean(
                torch.logsumexp(similarities, dim=1) - similarities.diagonal()
            )
            if with_regularizer:
                loss = loss + torch.mean(torch.sum(embeddings**2, dim=1))
            return loss

        for with_regularizer in (False, True):
            regularizer = LpRegularizer(power=2) if with_regularizer else None
            loss_func = NPairsLoss(embedding_regularizer=regularizer)
            for dtype in TEST_DTYPES:
                with self.subTest(dtype=dtype, regularizer=with_regularizer):
                    helpers.assert_matches_float64(
                        partial(loss_func, labels=labels),
                        partial(reference, with_regularizer=with_regularizer),
                        [coordinates],
                        dtype,
                    )
