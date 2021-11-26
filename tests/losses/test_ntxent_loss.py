import unittest

import torch

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import NTXentLoss, SupConLoss
from pytorch_metric_learning.reducers import AvgNonZeroReducer, PerAnchorReducer
from pytorch_metric_learning.utils import common_functions as c_f

from .. import TEST_DEVICE, TEST_DTYPES
from .utils import get_pair_embeddings_with_ref


class TestNTXentLoss(unittest.TestCase):
    def test_ntxent_loss(self):
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

            self.helper(embeddings, labels, pos_pairs, neg_pairs, dtype)

    def test_ntxent_loss_with_ref(self):
        for dtype in TEST_DTYPES:
            (
                embeddings,
                labels,
                ref_emb,
                ref_labels,
                pos_pairs,
                neg_pairs,
            ) = get_pair_embeddings_with_ref(dtype, TEST_DEVICE)
            self.helper(
                embeddings, labels, pos_pairs, neg_pairs, dtype, ref_emb, ref_labels
            )

    def helper(
        self,
        embeddings,
        labels,
        pos_pairs,
        neg_pairs,
        dtype,
        ref_emb=None,
        ref_labels=None,
    ):
        temperature = 0.1
        loss_funcA = NTXentLoss(temperature=temperature)
        loss_funcB = NTXentLoss(temperature=temperature, distance=LpDistance())
        loss_funcC = NTXentLoss(
            temperature=temperature, reducer=PerAnchorReducer(AvgNonZeroReducer())
        )
        loss_funcD = SupConLoss(temperature=temperature)
        loss_funcE = SupConLoss(temperature=temperature, distance=LpDistance())

        obtained_losses = [
            x(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
            for x in [loss_funcA, loss_funcB, loss_funcC, loss_funcD, loss_funcE]
        ]

        loss_len = 4 if ref_emb is not None else 5
        total_lossA, total_lossB, total_lossC, total_lossD, total_lossE = (
            0,
            0,
            torch.zeros(loss_len, device=TEST_DEVICE, dtype=dtype),
            torch.zeros(loss_len, device=TEST_DEVICE, dtype=dtype),
            torch.zeros(loss_len, device=TEST_DEVICE, dtype=dtype),
        )
        for a1, p in pos_pairs:
            anchor = embeddings[a1]
            positive = embeddings[p] if ref_emb is None else ref_emb[p]
            numeratorA = torch.exp(torch.matmul(anchor, positive) / temperature)
            numeratorB = torch.exp(
                -torch.sqrt(torch.sum((anchor - positive) ** 2)) / temperature
            )
            denominatorA = numeratorA.clone()
            denominatorB = numeratorB.clone()
            denominatorD = 0
            denominatorE = 0
            for a2, n in pos_pairs + neg_pairs:
                if a2 == a1:
                    negative = embeddings[n] if ref_emb is None else ref_emb[n]
                    curr_denomD = torch.exp(
                        torch.matmul(anchor, negative) / temperature
                    )
                    curr_denomE = torch.exp(
                        -torch.sqrt(torch.sum((anchor - negative) ** 2)) / temperature
                    )
                    denominatorD += curr_denomD
                    denominatorE += curr_denomE
                    if (a2, n) not in pos_pairs:
                        denominatorA += curr_denomD
                        denominatorB += curr_denomE
                else:
                    continue

            curr_lossA = -torch.log(numeratorA / denominatorA)
            curr_lossB = -torch.log(numeratorB / denominatorB)
            curr_lossD = -torch.log(numeratorA / denominatorD)
            curr_lossE = -torch.log(numeratorB / denominatorE)
            total_lossA += curr_lossA
            total_lossB += curr_lossB
            total_lossC[a1] += curr_lossA
            total_lossD[a1] += curr_lossD
            total_lossE[a1] += curr_lossE

        total_lossA /= len(pos_pairs)
        total_lossB /= len(pos_pairs)
        if ref_emb is not None:
            pos_pair_per_anchor = torch.tensor([1, 1, 1, 1])
        else:
            pos_pair_per_anchor = torch.tensor([2, 2, 2, 1, 1])
        pos_pair_per_anchor = pos_pair_per_anchor.to(TEST_DEVICE).type(dtype)
        total_lossC, total_lossD, total_lossE = [
            torch.mean(x / pos_pair_per_anchor)
            for x in [total_lossC, total_lossD, total_lossE]
        ]

        rtol = 1e-2 if dtype == torch.float16 else 1e-5
        self.assertTrue(torch.isclose(obtained_losses[0], total_lossA, rtol=rtol))
        self.assertTrue(torch.isclose(obtained_losses[1], total_lossB, rtol=rtol))
        self.assertTrue(torch.isclose(obtained_losses[2], total_lossC, rtol=rtol))
        self.assertTrue(torch.isclose(obtained_losses[3], total_lossD, rtol=rtol))
        self.assertTrue(torch.isclose(obtained_losses[4], total_lossE, rtol=rtol))

    def test_with_no_valid_pairs(self):
        all_embedding_angles = [[0], [0, 10, 20], [0, 40, 60]]
        all_labels = [
            torch.LongTensor([0]),
            torch.LongTensor([0, 0, 0]),
            torch.LongTensor([1, 2, 3]),
        ]
        temperature = 0.1
        for loss_class in [NTXentLoss, SupConLoss]:
            loss_funcA = loss_class(temperature)
            loss_funcB = loss_class(temperature, distance=LpDistance())
            for loss_func in [loss_funcA, loss_funcB]:
                for dtype in TEST_DTYPES:
                    for embedding_angles, labels in zip(
                        all_embedding_angles, all_labels
                    ):
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
        for loss_class in [NTXentLoss, SupConLoss]:
            loss_funcA = loss_class(temperature)
            loss_funcB = loss_class(temperature, distance=LpDistance())
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
