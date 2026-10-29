import unittest

import torch

from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.miners import MultiSimilarityMiner
from pytorch_metric_learning.utils import _common as c_f

from .. import TEST_DEVICE, TEST_DTYPES, helpers


def positive_and_negative_pairs(labels):
    num_points = len(labels)
    pairs = [(i, j) for i in range(num_points) for j in range(num_points) if j != i]
    positive_pairs = [(i, j) for i, j in pairs if labels[i] == labels[j]]
    negative_pairs = [(i, j) for i, j in pairs if labels[i] != labels[j]]
    return positive_pairs, negative_pairs


def hardest_negative_similarity(mat, negative_pairs, anchor, dtype, is_inverted):
    hardest = c_f.neg_inf(dtype) if is_inverted else c_f.pos_inf(dtype)
    for a, n in negative_pairs:
        if a != anchor:
            continue
        an_sim = mat[a, n]
        if (an_sim > hardest) if is_inverted else (an_sim < hardest):
            hardest = an_sim
    return hardest


def hardest_positive_similarity(mat, positive_pairs, anchor, dtype, is_inverted):
    hardest = c_f.pos_inf(dtype) if is_inverted else c_f.neg_inf(dtype)
    for a, p in positive_pairs:
        if a != anchor:
            continue
        ap_sim = mat[a, p]
        if (ap_sim < hardest) if is_inverted else (ap_sim > hardest):
            hardest = ap_sim
    return hardest


def expected_positive_pairs(mat, pairs, epsilon, dtype, is_inverted):
    positive_pairs, negative_pairs = pairs
    expected = set()
    for a, p in positive_pairs:
        hardest = hardest_negative_similarity(
            mat, negative_pairs, a, dtype, is_inverted
        )
        ap_sim = mat[a, p]
        if (
            (ap_sim < hardest + epsilon)
            if is_inverted
            else (ap_sim > hardest - epsilon)
        ):
            expected.add((a, p))
    return expected


def expected_negative_pairs(mat, pairs, epsilon, dtype, is_inverted):
    positive_pairs, negative_pairs = pairs
    expected = set()
    for a, n in negative_pairs:
        hardest = hardest_positive_similarity(
            mat, positive_pairs, a, dtype, is_inverted
        )
        an_sim = mat[a, n]
        if (
            (an_sim > hardest - epsilon)
            if is_inverted
            else (an_sim < hardest + epsilon)
        ):
            expected.add((a, n))
    return expected


def mined_pairs(anchors, others):
    return {(a.item(), o.item()) for a, o in zip(anchors, others, strict=True)}


class TestMultiSimilarityMiner(unittest.TestCase):
    def test_multi_similarity_miner(self):
        torch.manual_seed(9599)
        epsilon = 0.1
        for dtype in TEST_DTYPES:
            for distance in [CosineSimilarity(), LpDistance()]:
                miner = MultiSimilarityMiner(epsilon, distance=distance)
                embeddings = helpers.planar_embeddings(range(64), dtype)
                labels = torch.randint(low=0, high=10, size=(64,))
                mat = distance(embeddings)
                pairs = positive_and_negative_pairs(labels)
                correct_pos_pairs = expected_positive_pairs(
                    mat, pairs, epsilon, dtype, distance.is_inverted
                )
                correct_neg_pairs = expected_negative_pairs(
                    mat, pairs, epsilon, dtype, distance.is_inverted
                )

                a1, p1, a2, n2 = miner(embeddings, labels)
                self.assertTrue(mined_pairs(a1, p1) == correct_pos_pairs)
                self.assertTrue(mined_pairs(a2, n2) == correct_neg_pairs)

    def test_empty_output(self):
        miner = MultiSimilarityMiner(0.1)
        batch_size = 32
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(batch_size, 64).type(dtype).to(TEST_DEVICE)
            labels = torch.arange(batch_size)
            a1, p, _, _ = miner(embeddings, labels)
            self.assertTrue(len(a1) == 0)
            self.assertTrue(len(p) == 0)
