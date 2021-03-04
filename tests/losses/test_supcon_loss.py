import unittest

import torch
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses.supcon_loss import SupConLoss
from pytorch_metric_learning.utils import common_functions as c_f

from .. import TEST_DEVICE, TEST_DTYPES


class TestSupConLoss(unittest.TestCase):
    def test_sup_con_loss(self):
        temperature = 0.1
        loss_funcA = SupConLoss(temperature=temperature)
        loss_funcB = SupConLoss(temperature=temperature, distance=LpDistance())

        for dtype in TEST_DTYPES:
            embedding_angles = [0, 10, 20, 50, 60, 80]
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

            anchor_pos_pairs = {
                0: [(0, 1), (0, 2)],
                1: [(1, 0), (1, 2)],
                2: [(2, 0), (2, 1)],
                3: [(3, 4)],
                4: [(4, 3)],
            }
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

            total_lossA = 0
            total_lossB = 0
            for a1, pos_pairs in anchor_pos_pairs.items():
                anchor_lossA = 0
                anchor_lossB = 0
                for __, p in pos_pairs:
                    anchor, positive = embeddings[a1], embeddings[p]
                    numeratorA = torch.exp(torch.matmul(anchor, positive) / temperature)
                    numeratorB = torch.exp(
                        -torch.sqrt(torch.sum((anchor - positive) ** 2)) / temperature
                    )
                    denominatorA = 0
                    denominatorB = 0
                    for a2, n in pos_pairs + neg_pairs:
                        if a2 == a1:
                            negative = embeddings[n]
                        else:
                            continue
                        denominatorA += torch.exp(torch.matmul(anchor, negative) / temperature)
                        denominatorB += torch.exp(
                            -torch.sqrt(torch.sum((anchor - negative) ** 2)) / temperature
                        )
                    anchor_lossA += torch.log(numeratorA / denominatorA)
                    anchor_lossB += torch.log(numeratorB / denominatorB)
                total_lossA += -anchor_lossA / len(pos_pairs)
                total_lossB += -anchor_lossB / len(pos_pairs)
            total_lossA /= len(anchor_pos_pairs)
            total_lossB /= len(anchor_pos_pairs)
            rtol = 1e-2 if dtype == torch.float16 else 1e-5
            self.assertTrue(torch.isclose(lossA, total_lossA, rtol=rtol))
            self.assertTrue(torch.isclose(lossB, total_lossB, rtol=rtol))

    def test_with_no_valid_pairs(self):
        loss_func = SupConLoss(temperature=0.1)
        all_embedding_angles = [[0], [0, 10, 20], [0, 40, 60]]
        all_labels = [
            torch.LongTensor([0]),
            torch.LongTensor([0, 0, 0]),
            torch.LongTensor([1, 2, 3]),
        ]
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
        loss_funcA = SupConLoss(temperature=temperature)
        loss_funcB = SupConLoss(temperature=temperature, distance=LpDistance())
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
