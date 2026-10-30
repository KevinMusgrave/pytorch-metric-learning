import unittest

import torch

from pytorch_metric_learning.losses import (
    ContrastiveLoss,
    CrossBatchMemory,
    MultipleLosses,
    TripletMarginLoss,
)
from pytorch_metric_learning.miners import MultiSimilarityMiner
from pytorch_metric_learning.types import PairIndices
from pytorch_metric_learning.utils import indexing

from .. import TEST_DEVICE, TEST_DTYPES
from ..helpers import angle_to_coord


class TestMultipleLosses(unittest.TestCase):
    def test_multiple_losses(self):
        lossA = ContrastiveLoss()
        lossB = TripletMarginLoss(0.1)
        minerB = MultiSimilarityMiner()
        loss_func1 = MultipleLosses(
            losses={"lossA": lossA, "lossB": lossB},
            weights={"lossA": 1, "lossB": 0.23},
            miners={"lossB": minerB},
        )

        for loss_func in [loss_func1]:
            for dtype in TEST_DTYPES:
                embedding_angles = torch.arange(0, 180)
                embeddings = torch.tensor(
                    [angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(TEST_DEVICE)
                labels = torch.randint(low=0, high=10, size=(180,)).to(TEST_DEVICE)

                loss = loss_func(embeddings, labels)
                loss.backward()

                indices_tupleB = minerB(embeddings, labels)
                correct_loss = (
                    lossA(embeddings, labels)
                    + lossB(embeddings, labels, indices_tupleB) * 0.23
                )
                self.assertTrue(torch.isclose(loss, correct_loss))

    def test_miners_select_from_supplied_candidates(self):
        lossA = ContrastiveLoss()
        lossB = TripletMarginLoss(0.1)
        minerB = MultiSimilarityMiner()
        loss_func = MultipleLosses(
            losses={"lossA": lossA, "lossB": lossB},
            miners={"lossB": minerB},
            weights={"lossA": 1, "lossB": 0.23},
        )
        embeddings = torch.randn(32, 4, device=TEST_DEVICE, dtype=torch.float64)
        labels = torch.randint(0, 4, (32,), device=TEST_DEVICE)
        candidates = indexing.get_all_pairs_indices(labels)
        candidates = PairIndices(*(x[: len(x) // 2] for x in candidates))
        mined = minerB(embeddings, labels, candidates)
        expected = (
            lossA(embeddings, labels, candidates)
            + lossB(embeddings, labels, mined) * 0.23
        )
        self.assertNotEqual(
            lossB(embeddings, labels, mined).item(),
            lossB(embeddings, labels, minerB(embeddings, labels)).item(),
        )
        torch.testing.assert_close(loss_func(embeddings, labels, candidates), expected)

    def test_miners_refine_cross_batch_memory_pairs(self):
        torch.manual_seed(4)
        miner = MultiSimilarityMiner(epsilon=0.3)
        combined = CrossBatchMemory(
            MultipleLosses({"loss": ContrastiveLoss()}, miners={"loss": miner}),
            embedding_size=4,
            memory_size=32,
        ).to(TEST_DEVICE, torch.float64)
        direct = CrossBatchMemory(
            ContrastiveLoss(), embedding_size=4, memory_size=32, miner=miner
        ).to(TEST_DEVICE, torch.float64)
        for _ in range(3):
            embeddings = torch.randn(8, 4, device=TEST_DEVICE, dtype=torch.float64)
            labels = torch.randint(0, 3, (8,), device=TEST_DEVICE)
            torch.testing.assert_close(
                combined(embeddings, labels), direct(embeddings, labels)
            )

    def test_input_indices_tuple(self):
        lossA = ContrastiveLoss()
        lossB = TripletMarginLoss(0.1)
        miner = MultiSimilarityMiner()
        loss_func1 = MultipleLosses(
            losses={"lossA": lossA, "lossB": lossB}, weights={"lossA": 1, "lossB": 0.23}
        )

        for loss_func in [loss_func1]:
            for dtype in TEST_DTYPES:
                embedding_angles = torch.arange(0, 180)
                embeddings = torch.tensor(
                    [angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(TEST_DEVICE)
                labels = torch.randint(low=0, high=10, size=(180,)).to(TEST_DEVICE)
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
            ValueError,
            lambda: MultipleLosses(
                losses={"lossA": lossA, "lossB": lossB},
                weights={"blah": 1, "lossB": 0.23},
            ),
        )

        minerA = MultiSimilarityMiner()
        self.assertRaises(
            ValueError,
            lambda: MultipleLosses(
                losses={"lossA": lossA, "lossB": lossB},
                weights={"lossA": 1, "lossB": 0.23},
                miners={"blah": minerA},
            ),
        )
