import unittest

import pytest
import torch

from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.miners import MultiSimilarityMiner
from pytorch_metric_learning.utils import _common as c_f

from .. import TEST_DEVICE, TEST_DTYPES, helpers

REPORTED_LABELS = [15, 15, 15, 15, 169, 169, 169, 169]
REPORTED_SIMILARITIES = [
    [1.0000, 0.9996, 0.9975, 0.9994, 0.9948, 0.9836, 0.9968, 0.9975],
    [0.9996, 1.0000, 0.9952, 0.9981, 0.9919, 0.9798, 0.9950, 0.9963],
    [0.9975, 0.9952, 1.0000, 0.9991, 0.9977, 0.9879, 0.9993, 0.9980],
    [0.9994, 0.9981, 0.9991, 1.0000, 0.9974, 0.9876, 0.9979, 0.9975],
    [0.9948, 0.9919, 0.9977, 0.9974, 1.0000, 0.9960, 0.9947, 0.9917],
    [0.9836, 0.9798, 0.9879, 0.9876, 0.9960, 1.0000, 0.9823, 0.9767],
    [0.9968, 0.9950, 0.9993, 0.9979, 0.9947, 0.9823, 1.0000, 0.9992],
    [0.9975, 0.9963, 0.9980, 0.9975, 0.9917, 0.9767, 0.9992, 1.0000],
]


def mined_pair_sets(miner, mat, labels):
    embeddings = torch.zeros(len(labels), 2, device=mat.device, dtype=mat.dtype)
    a1, p, a2, n = miner(embeddings, labels, mat=mat)
    positives = sorted(zip(a1.tolist(), p.tolist(), strict=True))
    negatives = sorted(zip(a2.tolist(), n.tolist(), strict=True))
    return positives, negatives


def positive_and_negative_pairs(labels):
    num_points = len(labels)
    pairs = [(i, j) for i in range(num_points) for j in range(num_points) if j != i]
    positive_pairs = [(i, j) for i, j in pairs if labels[i] == labels[j]]
    negative_pairs = [(i, j) for i, j in pairs if labels[i] != labels[j]]
    return positive_pairs, negative_pairs


def hardest_negative_similarity(mat, negative_pairs, anchor, is_inverted):
    hardest = c_f.neg_inf(mat.dtype) if is_inverted else c_f.pos_inf(mat.dtype)
    for a, n in negative_pairs:
        if a != anchor:
            continue
        an_sim = mat[a, n].item()
        if (an_sim > hardest) if is_inverted else (an_sim < hardest):
            hardest = an_sim
    return hardest


def hardest_positive_similarity(mat, positive_pairs, anchor, is_inverted):
    hardest = c_f.pos_inf(mat.dtype) if is_inverted else c_f.neg_inf(mat.dtype)
    for a, p in positive_pairs:
        if a != anchor:
            continue
        ap_sim = mat[a, p].item()
        if (ap_sim < hardest) if is_inverted else (ap_sim > hardest):
            hardest = ap_sim
    return hardest


def positive_pair_margins(mat, pairs, epsilon, is_inverted):
    positive_pairs, negative_pairs = pairs
    margins = {}
    for a, p in positive_pairs:
        hardest = hardest_negative_similarity(mat, negative_pairs, a, is_inverted)
        ap_sim = mat[a, p].item()
        margins[(a, p)] = (
            hardest + epsilon - ap_sim if is_inverted else ap_sim - hardest + epsilon
        )
    return margins


def negative_pair_margins(mat, pairs, epsilon, is_inverted):
    positive_pairs, negative_pairs = pairs
    margins = {}
    for a, n in negative_pairs:
        hardest = hardest_positive_similarity(mat, positive_pairs, a, is_inverted)
        an_sim = mat[a, n].item()
        margins[(a, n)] = (
            an_sim - hardest + epsilon if is_inverted else hardest + epsilon - an_sim
        )
    return margins


def mined_pairs(anchors, others):
    return {(a.item(), o.item()) for a, o in zip(anchors, others, strict=True)}


def assert_mined_by_margin(mined, margins, dtype):
    tolerance = 4 * torch.finfo(dtype).eps
    decided = {pair for pair, margin in margins.items() if abs(margin) > tolerance}
    assert mined <= margins.keys()
    assert mined & decided == {pair for pair in decided if margins[pair] > 0}


class TestMultiSimilarityMiner(unittest.TestCase):
    def test_reported_similarities_select_only_valid_pairs(self):
        mat = torch.tensor(REPORTED_SIMILARITIES, device=TEST_DEVICE)
        labels = torch.tensor(REPORTED_LABELS, device=TEST_DEVICE)
        expected_positives, expected_negatives = positive_and_negative_pairs(
            REPORTED_LABELS
        )
        positives, negatives = mined_pair_sets(MultiSimilarityMiner(), mat, labels)
        self.assertEqual(positives, sorted(expected_positives))
        self.assertEqual(negatives, sorted(expected_negatives))

    @pytest.mark.mps
    def test_reported_similarities_agree_with_cpu_on_mps(self):
        mat = torch.tensor(REPORTED_SIMILARITIES)
        labels = torch.tensor(REPORTED_LABELS)
        miner = MultiSimilarityMiner()
        self.assertEqual(
            mined_pair_sets(miner, mat.to("mps"), labels.to("mps")),
            mined_pair_sets(miner, mat, labels),
        )

    @pytest.mark.mps
    def test_miner_agrees_with_cpu_on_mps_for_one_matrix(self):
        generator = torch.Generator().manual_seed(4)
        miner = MultiSimilarityMiner()
        for size, classes in ((8, 2), (16, 4), (64, 8), (256, 32)):
            for spread in (0.05, 1.0):
                with self.subTest(size=size, spread=spread):
                    rows = torch.ones(size, 32) + spread * torch.randn(
                        size, 32, generator=generator
                    )
                    labels = torch.arange(size) % classes
                    mat = CosineSimilarity()(rows)
                    on_cpu = mined_pair_sets(miner, mat, labels)
                    on_mps = mined_pair_sets(miner, mat.to("mps"), labels.to("mps"))
                    self.assertEqual(on_mps, on_cpu)
                    for a, p in on_mps[0]:
                        self.assertEqual(labels[a], labels[p])
                    for a, n in on_mps[1]:
                        self.assertNotEqual(labels[a], labels[n])

    def test_multi_similarity_miner(self):
        torch.manual_seed(9599)
        epsilon = 0.1
        for dtype in TEST_DTYPES:
            for distance in [CosineSimilarity(), LpDistance()]:
                miner = MultiSimilarityMiner(epsilon, distance=distance)
                embeddings = helpers.planar_embeddings(range(64), dtype)
                labels = torch.randint(low=0, high=10, size=(64,)).to(TEST_DEVICE)
                mat = distance(embeddings.detach().double())
                pairs = positive_and_negative_pairs(labels)

                a1, p1, a2, n2 = miner(embeddings, labels)
                assert_mined_by_margin(
                    mined_pairs(a1, p1),
                    positive_pair_margins(mat, pairs, epsilon, distance.is_inverted),
                    dtype,
                )
                assert_mined_by_margin(
                    mined_pairs(a2, n2),
                    negative_pair_margins(mat, pairs, epsilon, distance.is_inverted),
                    dtype,
                )

    def test_empty_output(self):
        miner = MultiSimilarityMiner(0.1)
        batch_size = 32
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(batch_size, 64).type(dtype).to(TEST_DEVICE)
            labels = torch.arange(batch_size).to(TEST_DEVICE)
            a1, p, _, _ = miner(embeddings, labels)
            self.assertTrue(len(a1) == 0)
            self.assertTrue(len(p) == 0)
