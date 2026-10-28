import unittest

import torch

from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.miners import PairMarginMiner

from .. import TEST_DEVICE, TEST_DTYPES
from ..helpers import angle_to_coord


def planar_embeddings(num_points, dtype):
    return torch.tensor(
        [angle_to_coord(a) for a in torch.arange(0, num_points)],
        requires_grad=True,
        dtype=dtype,
    ).to(TEST_DEVICE)


def positive_and_negative_pairs(labels):
    num_points = len(labels)
    pairs = [(i, j) for i in range(num_points) for j in range(num_points) if j != i]
    positive_pairs = [(i, j) for i, j in pairs if labels[i] == labels[j]]
    negative_pairs = [(i, j) for i, j in pairs if labels[i] != labels[j]]
    return positive_pairs, negative_pairs


def expected_positive_pairs(mat, positive_pairs, pos_margin, is_inverted):
    return {
        (i, j)
        for i, j in positive_pairs
        if ((mat[i, j] < pos_margin) if is_inverted else (mat[i, j] > pos_margin))
    }


def expected_negative_pairs(mat, negative_pairs, neg_margin, is_inverted):
    return {
        (i, j)
        for i, j in negative_pairs
        if ((mat[i, j] > neg_margin) if is_inverted else (mat[i, j] < neg_margin))
    }


def mined_pairs(anchors, others):
    return {(a.item(), o.item()) for a, o in zip(anchors, others, strict=True)}


class TestPairMarginMiner(unittest.TestCase):
    def test_pair_margin_miner(self):
        for dtype in TEST_DTYPES:
            for distance in [LpDistance(), CosineSimilarity()]:
                embeddings = planar_embeddings(16, dtype)
                labels = torch.randint(low=0, high=2, size=(16,))
                mat = distance(embeddings)
                pos_pairs, neg_pairs = positive_and_negative_pairs(labels)
                for pos_margin_int in range(-1, 4):
                    pos_margin = float(pos_margin_int) * 0.05
                    correct_pos = expected_positive_pairs(
                        mat, pos_pairs, pos_margin, distance.is_inverted
                    )
                    for neg_margin_int in range(2, 7):
                        neg_margin = float(neg_margin_int) * 0.05
                        correct_neg = expected_negative_pairs(
                            mat, neg_pairs, neg_margin, distance.is_inverted
                        )
                        miner = PairMarginMiner(
                            pos_margin, neg_margin, distance=distance
                        )
                        a1, p1, a2, n2 = miner(embeddings, labels)
                        self.assertTrue(mined_pairs(a1, p1) == correct_pos)
                        self.assertTrue(mined_pairs(a2, n2) == correct_neg)

    def test_empty_output(self):
        miner = PairMarginMiner(0, 1)
        batch_size = 32
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(batch_size, 64).type(dtype).to(TEST_DEVICE)
            labels = torch.arange(batch_size)
            a, p, _, _ = miner(embeddings, labels)
            self.assertTrue(len(a) == 0)
            self.assertTrue(len(p) == 0)
