import unittest
from .. import TEST_DTYPES
import torch
from pytorch_metric_learning.miners import BatchHardMiner
from pytorch_metric_learning.utils import common_functions as c_f
import numpy as np
from pytorch_metric_learning.distances import CosineSimilarity, LpDistance


class TestBatchHardMiner(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.device = torch.device("cuda")
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
        self.correct_a = torch.LongTensor([0, 1, 2, 3, 4, 6, 7, 8]).to(self.device)
        self.correct_p = torch.LongTensor([4, 4, 8, 8, 0, 2, 2, 2]).to(self.device)
        self.correct_n = [
            torch.LongTensor([2, 2, 1, 4, 3, 5, 5, 5]).to(self.device),
            torch.LongTensor([2, 2, 1, 4, 5, 5, 5, 5]).to(self.device),
        ]

    def test_dist_mining(self):
        for dtype in TEST_DTYPES:
            embeddings = torch.arange(9).type(dtype).unsqueeze(1).to(self.device)
            a, p, n = self.dist_miner(embeddings, self.labels)
            self.helper(a, p, n)
            self.assertTrue(self.dist_miner.hardest_pos_pair_dist == 6)
            self.assertTrue(self.dist_miner.hardest_neg_pair_dist == 1)

    def test_normalized_dist_mining(self):
        for dtype in TEST_DTYPES:
            angles = [0, 20, 40, 60, 80, 100, 120, 140, 160]
            embeddings = torch.tensor(
                [c_f.angle_to_coord(a) for a in angles], dtype=dtype
            ).to(self.device)
            a, p, n = self.normalized_dist_miner(embeddings, self.labels)
            self.helper(a, p, n)
            correct_hardest_pos_pair_dist = torch.sqrt(
                torch.sum((embeddings[2] - embeddings[8]) ** 2)
            ).item()
            correct_hardest_neg_pair_dist = torch.sqrt(
                torch.sum((embeddings[1] - embeddings[2]) ** 2)
            ).item()
            places = 2 if dtype == torch.float16 else 5
            self.assertAlmostEqual(
                self.normalized_dist_miner.hardest_pos_pair_dist,
                correct_hardest_pos_pair_dist,
                places=places,
            )
            self.assertAlmostEqual(
                self.normalized_dist_miner.hardest_neg_pair_dist,
                correct_hardest_neg_pair_dist,
                places=places,
            )

    def test_normalized_dist_squared_mining(self):
        for dtype in TEST_DTYPES:
            angles = [0, 20, 40, 60, 80, 100, 120, 140, 160]
            embeddings = torch.tensor(
                [c_f.angle_to_coord(a) for a in angles], dtype=dtype
            ).to(self.device)
            a, p, n = self.normalized_dist_miner_squared(embeddings, self.labels)
            self.helper(a, p, n)
            correct_hardest_pos_pair_dist = torch.sum(
                (embeddings[2] - embeddings[8]) ** 2
            ).item()
            correct_hardest_neg_pair_dist = torch.sum(
                (embeddings[1] - embeddings[2]) ** 2
            ).item()
            places = 2 if dtype == torch.float16 else 5
            self.assertAlmostEqual(
                self.normalized_dist_miner_squared.hardest_pos_pair_dist,
                correct_hardest_pos_pair_dist,
                places=places,
            )
            self.assertAlmostEqual(
                self.normalized_dist_miner_squared.hardest_neg_pair_dist,
                correct_hardest_neg_pair_dist,
                places=places,
            )

    def test_sim_mining(self):
        for dtype in TEST_DTYPES:
            angles = [0, 20, 40, 60, 80, 100, 120, 140, 160]
            embeddings = torch.tensor(
                [c_f.angle_to_coord(a) for a in angles], dtype=dtype
            ).to(self.device)
            a, p, n = self.sim_miner(embeddings, self.labels)
            self.helper(a, p, n)
            places = 2 if dtype == torch.float16 else 5
            self.assertAlmostEqual(
                self.sim_miner.hardest_pos_pair_dist,
                np.cos(np.radians(120)),
                places=places,
            )
            self.assertAlmostEqual(
                self.sim_miner.hardest_neg_pair_dist,
                np.cos(np.radians(20)),
                places=places,
            )

    def helper(self, a, p, n):
        self.assertTrue(torch.equal(a, self.correct_a))
        self.assertTrue(torch.equal(p, self.correct_p))
        self.assertTrue(any(torch.equal(n, cn) for cn in self.correct_n))

    def test_empty_output(self):
        batch_size = 32
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(batch_size, 64).type(dtype).to(self.device)
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
                self.assertTrue(miner.hardest_pos_pair_dist == 0)
                self.assertTrue(miner.hardest_neg_pair_dist == 0)
                self.assertTrue(miner.hardest_triplet_dist == 0)
