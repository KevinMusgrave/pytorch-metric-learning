import unittest

import torch

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.reducers import PerAnchorReducer
from pytorch_metric_learning.utils import common_functions as c_f

from .. import TEST_DEVICE, TEST_DTYPES


class TestNTXentLoss(unittest.TestCase):
    def test_ntxent_loss(self):
        temperature = 0.1
        loss_funcA = NTXentLoss(temperature=temperature)
        loss_funcB = NTXentLoss(temperature=temperature, distance=LpDistance())
        loss_funcC = NTXentLoss(temperature=temperature, reducer=PerAnchorReducer())

        for dtype in TEST_DTYPES:
            embedding_angles = [0, 20, 40, 60, 80, 100]
            embeddings = torch.tensor(
                [c_f.angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings

            labels = torch.LongTensor([0, 0, 0, 1, 1, 2])

            lossA = loss_funcA(embeddings, labels)
            lossB = loss_funcB(embeddings, labels)
            lossC = loss_funcC(embeddings, labels)

            pos_pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (3, 4), (4, 3)]
            neg_pairs = [
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 3),
                (1, 4),
                (1, 5),
                (2, 3),
                (2, 4),
                (2, 5),
                (3, 0),
                (3, 1),
                (3, 2),
                (3, 5),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 5),
                (5, 0),
                (5, 1),
                (5, 2),
                (5, 3),
                (5, 4),
            ]

            total_lossA, total_lossB, total_lossC = (
                0,
                0,
                torch.zeros(6, device=TEST_DEVICE, dtype=dtype),
            )
            for a1, p in pos_pairs:
                anchor, positive = embeddings[a1], embeddings[p]
                numeratorA = torch.exp(torch.matmul(anchor, positive) / temperature)
                numeratorB = torch.exp(
                    -torch.sqrt(torch.sum((anchor - positive) ** 2)) / temperature
                )
                denominatorA = numeratorA.clone()
                denominatorB = numeratorB.clone()
                for a2, n in neg_pairs:
                    if a2 == a1:
                        negative = embeddings[n]
                    else:
                        continue
                    denominatorA += torch.exp(
                        torch.matmul(anchor, negative) / temperature
                    )
                    denominatorB += torch.exp(
                        -torch.sqrt(torch.sum((anchor - negative) ** 2)) / temperature
                    )
                curr_lossA = -torch.log(numeratorA / denominatorA)
                curr_lossB = -torch.log(numeratorB / denominatorB)
                total_lossA += curr_lossA
                total_lossB += curr_lossB
                total_lossC[a1] += curr_lossA

            total_lossA /= len(pos_pairs)
            total_lossB /= len(pos_pairs)
            pos_pair_per_anchor = torch.tensor(
                [2, 2, 2, 1, 1, 0], device=TEST_DEVICE, dtype=dtype
            )
            total_lossC = total_lossC / pos_pair_per_anchor
            total_lossC[pos_pair_per_anchor == 0] = 0

            total_lossC = torch.mean(total_lossC)

            rtol = 1e-2 if dtype == torch.float16 else 1e-5
            self.assertTrue(torch.isclose(lossA, total_lossA, rtol=rtol))
            self.assertTrue(torch.isclose(lossB, total_lossB, rtol=rtol))
            self.assertTrue(torch.isclose(lossC, total_lossC, rtol=rtol))

    def test_with_no_valid_pairs(self):
        loss_func = NTXentLoss(temperature=0.1)
        all_embedding_angles = [[0], [0, 10, 20]]
        all_labels = [torch.LongTensor([0]), torch.LongTensor([0, 0, 0])]
        for dtype in TEST_DTYPES:
            for embedding_angles, labels in zip(all_embedding_angles, all_labels):
                embeddings = torch.tensor(
                    [c_f.angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(
                    TEST_DEVICE
                )  # 2D embeddings
                loss = loss_func(embeddings, labels)
                loss.backward()
                self.assertEqual(loss, 0)

    def test_backward(self):
        temperature = 0.1
        loss_funcA = NTXentLoss(temperature=temperature)
        loss_funcB = NTXentLoss(temperature=temperature, distance=LpDistance())
        for dtype in TEST_DTYPES:
            for loss_func in [loss_funcA, loss_funcB]:
                embedding_angles = [0, 20, 40, 60, 80]
                embeddings = torch.tensor(
                    [c_f.angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(
                    TEST_DEVICE
                )  # 2D embeddings
                labels = torch.LongTensor([0, 0, 1, 1, 2])
                loss = loss_func(embeddings, labels)
                loss.backward()
