import unittest

import torch

from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.miners import TripletMarginMiner

from .. import TEST_DEVICE, TEST_DTYPES, helpers

TRIPLET_TYPES = ("all", "hard", "semihard", "easy")


def anchor_positive_negative_triplets(labels):
    num_points = len(labels)
    for i in range(num_points):
        for j in range(num_points):
            if j == i or labels[j] != labels[i]:
                continue
            for k in range(num_points):
                if k not in (i, j) and labels[k] != labels[j]:
                    yield i, j, k


def triplet_distance_differences(mat, labels, is_inverted):
    differences = []
    for i, j, k in anchor_positive_negative_triplets(labels):
        ap_dist, an_dist = mat[i, j], mat[i, k]
        difference = ap_dist - an_dist if is_inverted else an_dist - ap_dist
        differences.append((i, j, k, difference))
    return differences


def expected_triplets_by_type(differences, margin):
    expected = {triplet_type: set() for triplet_type in TRIPLET_TYPES}
    for i, j, k, difference in differences:
        if difference > margin:
            expected["easy"].add((i, j, k))
            continue
        expected["all"].add((i, j, k))
        if difference > 0:
            expected["semihard"].add((i, j, k))
        if difference <= 0:
            expected["hard"].add((i, j, k))
    return expected


class TestTripletMarginMiner(unittest.TestCase):
    def test_triplet_margin_miner(self):
        for dtype in TEST_DTYPES:
            for distance in [LpDistance(), CosineSimilarity()]:
                embeddings = helpers.planar_embeddings(range(16), dtype)
                labels = torch.randint(low=0, high=2, size=(16,)).to(TEST_DEVICE)
                differences = triplet_distance_differences(
                    distance(embeddings), labels, distance.is_inverted
                )
                for margin_int in range(-1, 11):
                    margin = float(margin_int) * 0.05
                    expected = expected_triplets_by_type(differences, margin)
                    for triplet_type in TRIPLET_TYPES:
                        miner = TripletMarginMiner(
                            margin, type_of_triplets=triplet_type, distance=distance
                        )
                        self.assert_mines(
                            miner, embeddings, labels, expected[triplet_type]
                        )

    def assert_mines(self, miner, embeddings, labels, correct_triplets):
        a1, p1, n1 = miner(embeddings, labels)
        mined_triplets = {
            (a.item(), p.item(), n.item()) for a, p, n in zip(a1, p1, n1, strict=True)
        }
        self.assertTrue(mined_triplets == correct_triplets)

    def test_empty_output(self):
        miner = TripletMarginMiner(0.5, type_of_triplets="all")
        for dtype in TEST_DTYPES:
            batch_size = 32
            embeddings = torch.randn(batch_size, 64).type(dtype).to(TEST_DEVICE)
            labels = torch.arange(batch_size).to(TEST_DEVICE)
            a, p, n = miner(embeddings, labels)
            self.assertTrue(len(a) == 0)
            self.assertTrue(len(p) == 0)
            self.assertTrue(len(n) == 0)
