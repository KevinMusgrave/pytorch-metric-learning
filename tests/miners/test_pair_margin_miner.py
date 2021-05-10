import unittest

import torch

from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.miners import PairMarginMiner
from pytorch_metric_learning.utils import common_functions as c_f

from .. import TEST_DEVICE, TEST_DTYPES


class TestPairMarginMiner(unittest.TestCase):
    def test_pair_margin_miner(self):
        for dtype in TEST_DTYPES:
            for distance in [LpDistance(), CosineSimilarity()]:
                embedding_angles = torch.arange(0, 16)
                embeddings = torch.tensor(
                    [c_f.angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(
                    TEST_DEVICE
                )  # 2D embeddings
                labels = torch.randint(low=0, high=2, size=(16,))
                mat = distance(embeddings)
                pos_pairs = []
                neg_pairs = []
                for i in range(len(embeddings)):
                    anchor_label = labels[i]
                    for j in range(len(embeddings)):
                        if j == i:
                            continue
                        positive_label = labels[j]
                        if positive_label == anchor_label:
                            ap_dist = mat[i, j]
                            pos_pairs.append((i, j, ap_dist))

                for i in range(len(embeddings)):
                    anchor_label = labels[i]
                    for j in range(len(embeddings)):
                        if j == i:
                            continue
                        negative_label = labels[j]
                        if negative_label != anchor_label:
                            an_dist = mat[i, j]
                            neg_pairs.append((i, j, an_dist))

                for pos_margin_int in range(-1, 4):
                    pos_margin = float(pos_margin_int) * 0.05
                    for neg_margin_int in range(2, 7):
                        neg_margin = float(neg_margin_int) * 0.05
                        miner = PairMarginMiner(
                            pos_margin, neg_margin, distance=distance
                        )
                        correct_pos_pairs = []
                        correct_neg_pairs = []
                        for i, j, k in pos_pairs:
                            condition = (
                                (k < pos_margin)
                                if distance.is_inverted
                                else (k > pos_margin)
                            )
                            if condition:
                                correct_pos_pairs.append((i, j))
                        for i, j, k in neg_pairs:
                            condition = (
                                (k > neg_margin)
                                if distance.is_inverted
                                else (k < neg_margin)
                            )
                            if condition:
                                correct_neg_pairs.append((i, j))

                        correct_pos = set(correct_pos_pairs)
                        correct_neg = set(correct_neg_pairs)
                        a1, p1, a2, n2 = miner(embeddings, labels)
                        mined_pos = set([(a.item(), p.item()) for a, p in zip(a1, p1)])
                        mined_neg = set([(a.item(), n.item()) for a, n in zip(a2, n2)])

                        self.assertTrue(mined_pos == correct_pos)
                        self.assertTrue(mined_neg == correct_neg)

    def test_empty_output(self):
        miner = PairMarginMiner(0, 1)
        batch_size = 32
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(batch_size, 64).type(dtype).to(TEST_DEVICE)
            labels = torch.arange(batch_size)
            a, p, _, _ = miner(embeddings, labels)
            self.assertTrue(len(a) == 0)
            self.assertTrue(len(p) == 0)
