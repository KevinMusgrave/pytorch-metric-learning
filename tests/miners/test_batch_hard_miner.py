import unittest

import torch

from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.miners import BatchHardMiner

from .. import TEST_DEVICE, TEST_DTYPES
from ..zzz_testing_utils.testing_utils import angle_to_coord


class TestBatchHardMiner(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.dist_miner = BatchHardMiner(
            distance=LpDistance(normalize_embeddings=False)
        )
        self.normalized_dist_miner = BatchHardMiner(
            distance=LpDistance(normalize_embeddings=True)
        )
        self.normalized_dist_miner_squared = BatchHardMiner(
            distance=LpDistance(normalize_embeddings=True, power=2)
        )
        self.sim_miner = BatchHardMiner(distance=CosineSimilarity())
        self.labels = torch.LongTensor([0, 0, 1, 1, 0, 2, 1, 1, 1])
        self.correct_a = torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8]).to(TEST_DEVICE)
        self.correct_p = torch.LongTensor([4, 4, 8, 8, 0, 2, 2, 2]).to(TEST_DEVICE)
        self.correct_n = [
            torch.LongTensor([2, 2, 1, 4, 3, 5, 5, 5]).to(TEST_DEVICE),
            torch.LongTensor([2, 2, 1, 4, 5, 5, 5, 5]).to(TEST_DEVICE),
        ]

    @classmethod
    def tearDown(self):
        torch.cuda.empty_cache()

    def test_dist_mining(self):
        for dtype in TEST_DTYPES:
            embeddings = torch.arange(9).type(dtype).unsqueeze(1).to(TEST_DEVICE)
            a, p, n = self.dist_miner(embeddings, self.labels)
            self.helper(a, p, n)

    def test_normalized_dist_mining(self):
        for dtype in TEST_DTYPES:
            angles = [0, 20, 40, 60, 80, 100, 120, 140, 160]
            embeddings = torch.tensor(
                [angle_to_coord(a, normalized=True) for a in angles], dtype=dtype
            ).to(TEST_DEVICE)
            a, p, n = self.normalized_dist_miner(embeddings, self.labels)
            self.helper(a, p, n)

    def test_normalized_dist_squared_mining(self):
        for dtype in TEST_DTYPES:
            angles = [0, 20, 40, 60, 80, 100, 120, 140, 160]
            embeddings = torch.tensor(
                [angle_to_coord(a, normalized=True) for a in angles], dtype=dtype
            ).to(TEST_DEVICE)
            a, p, n = self.normalized_dist_miner_squared(embeddings, self.labels)
            self.helper(a, p, n)

    def test_sim_mining(self):
        for dtype in TEST_DTYPES:
            angles = [0, 20, 40, 60, 80, 100, 120, 140, 160]
            embeddings = torch.tensor(
                [angle_to_coord(a, normalized=True) for a in angles], dtype=dtype
            ).to(TEST_DEVICE)
            a, p, n = self.sim_miner(embeddings, self.labels)
            self.helper(a, p, n)

    def helper(self, a, p, n):
        self.assertTrue(torch.equal(a, self.correct_a))
        self.assertTrue(torch.equal(p, self.correct_p))
        self.assertTrue(any(torch.equal(n, cn) for cn in self.correct_n))

    def test_empty_output(self):
        batch_size = 32
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(batch_size, 64).type(dtype).to(TEST_DEVICE)
            labels = torch.arange(batch_size)
            for miner in [
                self.dist_miner,
                self.normalized_dist_miner,
                self.normalized_dist_miner_squared,
                self.sim_miner,
            ]:
                a, p, n = miner(embeddings, labels)
                self.assertTrue(len(a) == 0)
                self.assertTrue(len(p) == 0)
                self.assertTrue(len(n) == 0)
