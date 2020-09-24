import unittest
from .. import TEST_DTYPES
import torch
from pytorch_metric_learning.losses import NPairsLoss
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning.utils import common_functions as c_f


class TestNPairsLoss(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.device = torch.device("cuda")

    def test_npairs_loss(self):
        loss_funcA = NPairsLoss()
        loss_funcB = NPairsLoss(embedding_regularizer=LpRegularizer())

        for dtype in TEST_DTYPES:
            embedding_angles = list(range(0, 180, 20))[:7]
            embeddings = torch.tensor(
                [c_f.angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                self.device
            )  # 2D embeddings
            labels = torch.LongTensor([0, 0, 1, 1, 1, 2, 3])

            lossA = loss_funcA(embeddings, labels)
            lossB = loss_funcB(embeddings, labels)

            pos_pairs = [(0, 1), (2, 3)]
            neg_pairs = [(0, 3), (2, 1)]

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
                torch.isclose(lossB, total_loss + 1)
            )  # l2_reg is going to be 1 since the embeddings are normalized

    def test_with_no_valid_pairs(self):
        for dtype in TEST_DTYPES:
            loss_func = NPairsLoss()
            embedding_angles = [0]
            embeddings = torch.tensor(
                [c_f.angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                self.device
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
                    [c_f.angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(
                    self.device
                )  # 2D embeddings
                labels = torch.LongTensor([0, 0, 1, 1, 1, 2, 3])

                loss = loss_func(embeddings, labels)
                loss.backward()
