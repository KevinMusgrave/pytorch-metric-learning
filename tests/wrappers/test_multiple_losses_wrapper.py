import unittest

import torch

from pytorch_metric_learning.losses import (
    ContrastiveLoss,
    MultipleLosses,
    TripletMarginLoss,
)
from pytorch_metric_learning.miners import MultiSimilarityMiner

from .. import TEST_DEVICE, TEST_DTYPES
from ..zzz_testing_utils.testing_utils import angle_to_coord


class TestMultipleLossesWrapper(unittest.TestCase):
    def test_multiple_losses(self):
        lossA = ContrastiveLoss()
        lossB = TripletMarginLoss(0.1)
        minerB = MultiSimilarityMiner()
        loss_func1 = MultipleLosses(
            losses={"lossA": lossA, "lossB": lossB},
            weights={"lossA": 1, "lossB": 0.23},
            miners={"lossB": minerB},
        )

        loss_func2 = MultipleLosses(
            losses=[lossA, lossB], weights=[1, 0.23], miners=[None, minerB]
        )

        for loss_func in [loss_func1, loss_func2]:
            for dtype in TEST_DTYPES:
                embedding_angles = torch.arange(0, 180)
                embeddings = torch.tensor(
                    [angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(
                    TEST_DEVICE
                )  # 2D embeddings
                labels = torch.randint(low=0, high=10, size=(180,))

                loss = loss_func(embeddings, labels)
                loss.backward()

                indices_tupleB = minerB(embeddings, labels)
                correct_loss = (
                    lossA(embeddings, labels)
                    + lossB(embeddings, labels, indices_tupleB) * 0.23
                )
                self.assertTrue(torch.isclose(loss, correct_loss))
                self.assertRaises(
                    AssertionError,
                    lambda: loss_func(embeddings, labels, indices_tupleB),
                )

    def test_input_indices_tuple(self):
        lossA = ContrastiveLoss()
        lossB = TripletMarginLoss(0.1)
        miner = MultiSimilarityMiner()
        loss_func1 = MultipleLosses(
            losses={"lossA": lossA, "lossB": lossB}, weights={"lossA": 1, "lossB": 0.23}
        )

        loss_func2 = MultipleLosses(losses=[lossA, lossB], weights=[1, 0.23])

        for loss_func in [loss_func1, loss_func2]:
            for dtype in TEST_DTYPES:
                embedding_angles = torch.arange(0, 180)
                embeddings = torch.tensor(
                    [angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(
                    TEST_DEVICE
                )  # 2D embeddings
                labels = torch.randint(low=0, high=10, size=(180,))
                indices_tuple = miner(embeddings, labels)

                loss = loss_func(embeddings, labels, indices_tuple)
                loss.backward()

                correct_loss = (
                    lossA(embeddings, labels, indices_tuple)
                    + lossB(embeddings, labels, indices_tuple) * 0.23
                )
                self.assertTrue(torch.isclose(loss, correct_loss))

    def test_key_mismatch(self):
        lossA = ContrastiveLoss()
        lossB = TripletMarginLoss(0.1)
        self.assertRaises(
            AssertionError,
            lambda: MultipleLosses(
                losses={"lossA": lossA, "lossB": lossB},
                weights={"blah": 1, "lossB": 0.23},
            ),
        )

        minerA = MultiSimilarityMiner()
        self.assertRaises(
            AssertionError,
            lambda: MultipleLosses(
                losses={"lossA": lossA, "lossB": lossB},
                weights={"lossA": 1, "lossB": 0.23},
                miners={"blah": minerA},
            ),
        )

    def test_length_mistmatch(self):
        lossA = ContrastiveLoss()
        lossB = TripletMarginLoss(0.1)
        self.assertRaises(
            AssertionError,
            lambda: MultipleLosses(losses=[lossA, lossB], weights=[1]),
        )

        minerA = MultiSimilarityMiner()
        self.assertRaises(
            AssertionError,
            lambda: MultipleLosses(
                losses=[lossA, lossB],
                weights=[1, 0.2],
                miners=[minerA],
            ),
        )
