import unittest

import torch

from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.miners import HDCMiner

from .. import TEST_DEVICE, TEST_DTYPES
from ..zzz_testing_utils.testing_utils import angle_to_coord


class TestHDCMiner(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.dist_miner = HDCMiner(
            filter_percentage=0.3, distance=LpDistance(normalize_embeddings=False)
        )
        self.normalized_dist_miner = HDCMiner(
            filter_percentage=0.3, distance=LpDistance(normalize_embeddings=True)
        )
        self.normalized_dist_miner_squared = HDCMiner(
            filter_percentage=0.3,
            distance=LpDistance(normalize_embeddings=True, power=2),
        )
        self.sim_miner = HDCMiner(filter_percentage=0.3, distance=CosineSimilarity())
        self.labels = torch.LongTensor([0, 0, 1, 1, 1, 0])
        correct_a1 = torch.LongTensor([0, 5, 1, 5])
        correct_p = torch.LongTensor([5, 0, 5, 1])
        self.correct_pos_pairs = torch.stack([correct_a1, correct_p], dim=1).to(
            TEST_DEVICE
        )
        correct_a2 = torch.LongTensor([1, 2, 4, 5, 0, 2])
        correct_n = torch.LongTensor([2, 1, 5, 4, 2, 0])
        self.correct_neg_pairs = torch.stack([correct_a2, correct_n], dim=1).to(
            TEST_DEVICE
        )

    @classmethod
    def tearDown(self):
        torch.cuda.empty_cache()

    def test_dist_mining(self):
        for dtype in TEST_DTYPES:
            embeddings = torch.arange(6).type(dtype).to(TEST_DEVICE).unsqueeze(1)
            a1, p, a2, n = self.dist_miner(embeddings, self.labels)
            pos_pairs = torch.stack([a1, p], dim=1)
            neg_pairs = torch.stack([a2, n], dim=1)
            self.helper(pos_pairs, neg_pairs)

    def test_normalized_dist_mining(self):
        for dtype in TEST_DTYPES:
            angles = [0, 20, 40, 60, 80, 100]
            embeddings = torch.tensor(
                [angle_to_coord(a, normalized=True) for a in angles], dtype=dtype
            ).to(TEST_DEVICE)
            a1, p, a2, n = self.normalized_dist_miner(embeddings, self.labels)
            pos_pairs = torch.stack([a1, p], dim=1)
            neg_pairs = torch.stack([a2, n], dim=1)
            self.helper(pos_pairs, neg_pairs)

    def test_normalized_dist_squared_mining(self):
        for dtype in TEST_DTYPES:
            angles = [0, 20, 40, 60, 80, 100]
            embeddings = torch.tensor(
                [angle_to_coord(a, normalized=True) for a in angles], dtype=dtype
            ).to(TEST_DEVICE)
            a1, p, a2, n = self.normalized_dist_miner_squared(embeddings, self.labels)
            pos_pairs = torch.stack([a1, p], dim=1)
            neg_pairs = torch.stack([a2, n], dim=1)
            self.helper(pos_pairs, neg_pairs)

    def test_sim_mining(self):
        for dtype in TEST_DTYPES:
            angles = [0, 20, 40, 60, 80, 100]
            embeddings = torch.tensor(
                [angle_to_coord(a, normalized=True) for a in angles], dtype=dtype
            ).to(TEST_DEVICE)
            a1, p, a2, n = self.sim_miner(embeddings, self.labels)
            pos_pairs = torch.stack([a1, p], dim=1)
            neg_pairs = torch.stack([a2, n], dim=1)
            self.helper(pos_pairs, neg_pairs)

    def helper(self, pos_pairs, neg_pairs):
        self.assertTrue(len(pos_pairs) == 4)
        self.assertTrue(len(neg_pairs) == 6)

        diffs = pos_pairs[:, 0] - pos_pairs[:, 1]
        correct_diffs = self.correct_pos_pairs[:, 0] - self.correct_pos_pairs[:, 1]
        self.assertTrue(torch.equal(torch.abs(diffs), torch.abs(correct_diffs)))

        diffs = neg_pairs[:, 0] - neg_pairs[:, 1]
        correct_diffs = self.correct_neg_pairs[:, 0] - self.correct_neg_pairs[:, 1]
        self.assertTrue(torch.equal(torch.abs(diffs), torch.abs(correct_diffs)))

    def test_empty_output(self):
        for dtype in TEST_DTYPES:
            batch_size = 32
            embeddings = torch.randn(batch_size, 64).type(dtype).to(TEST_DEVICE)
            labels = torch.arange(batch_size)
            for miner in [
                self.dist_miner,
                self.normalized_dist_miner,
                self.normalized_dist_miner_squared,
                self.sim_miner,
            ]:
                a1, p, _, _ = miner(embeddings, labels)
                self.assertTrue(len(a1) == 0)
                self.assertTrue(len(p) == 0)


class TestHDCMinerCandidates(unittest.TestCase):
    def test_candidates_restrict_selection_without_state(self):
        torch.manual_seed(2)
        embeddings = torch.randn(8, 3, device=TEST_DEVICE)
        labels = torch.arange(4, device=TEST_DEVICE).repeat_interleave(2)
        miner = HDCMiner(filter_percentage=0.5)
        everything = miner(embeddings, labels)
        candidates = tuple(x[:4] for x in everything)
        restricted = miner(embeddings, labels, candidates=candidates)
        for pairs, (anchors, others) in (
            ((candidates[0], candidates[1]), restricted[:2]),
            ((candidates[2], candidates[3]), restricted[2:]),
        ):
            allowed = set(zip(pairs[0].tolist(), pairs[1].tolist(), strict=True))
            chosen = set(zip(anchors.tolist(), others.tolist(), strict=True))
            self.assertLessEqual(chosen, allowed)
            self.assertEqual(len(chosen), 2)
        again = miner(embeddings, labels)
        for first, second in zip(everything, again, strict=True):
            self.assertTrue(torch.equal(first, second))
