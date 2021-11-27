import unittest

import torch

from pytorch_metric_learning.losses import NPairsLoss
from pytorch_metric_learning.regularizers import LpRegularizer
from ..zzz_testing_utils.testing_utils import angle_to_coord

from .. import TEST_DEVICE, TEST_DTYPES


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
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            embeddings_for_loss_fn = (
                embeddings * embedding_norm
            )  # scale by random amount to test l2 regularizer
            labels = torch.LongTensor([0, 0, 1, 1, 1, 2, 3])

            lossA = loss_funcA(embeddings_for_loss_fn, labels)
            lossB = loss_funcB(embeddings_for_loss_fn, labels)

            pos_pairs = [(0, 1), (2, 3)]
            neg_pairs = [(0, 3), (2, 1)]

            embeddings = torch.nn.functional.normalize(embeddings)
            total_loss = 0
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
                torch.isclose(lossB, total_loss + torch.mean(torch.norm(embeddings_for_loss_fn, dim=1) ** 2))
            )  # l2_reg is going to be embedding_norm ** self.power

    def test_with_no_valid_pairs(self):
        for dtype in TEST_DTYPES:
            loss_func = NPairsLoss()
            embedding_angles = [0]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.LongTensor([0])
            loss = loss_func(embeddings, labels)
            loss.backward()
            self.assertEqual(loss, 0)

    def test_backward(self):
        loss_funcA = NPairsLoss()
        loss_funcB = NPairsLoss(embedding_regularizer=LpRegularizer())

        for dtype in TEST_DTYPES:
            for loss_func in [loss_funcA, loss_funcB]:
                embedding_angles = list(range(0, 180, 20))[:7]
                embeddings = torch.tensor(
                    [angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(
                    TEST_DEVICE
                )  # 2D embeddings
                labels = torch.LongTensor([0, 0, 1, 1, 1, 2, 3])

                loss = loss_func(embeddings, labels)
                loss.backward()
