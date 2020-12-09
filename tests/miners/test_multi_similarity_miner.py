import unittest
from .. import TEST_DTYPES, TEST_DEVICE
import torch
from pytorch_metric_learning.miners import MultiSimilarityMiner
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity


class TestMultiSimilarityMiner(unittest.TestCase):
    def test_multi_similarity_miner(self):
        epsilon = 0.1
        for dtype in TEST_DTYPES:
            for distance in [CosineSimilarity(), LpDistance()]:
                miner = MultiSimilarityMiner(epsilon, distance=distance)
                embedding_angles = torch.arange(0, 64)
                embeddings = torch.tensor(
                    [c_f.angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(
                    TEST_DEVICE
                )  # 2D embeddings
                labels = torch.randint(low=0, high=10, size=(64,))
                mat = distance(embeddings)
                pos_pairs = []
                neg_pairs = []
                for i in range(len(embeddings)):
                    anchor_label = labels[i]
                    for j in range(len(embeddings)):
                        if j != i:
                            other_label = labels[j]
                            if anchor_label == other_label:
                                pos_pairs.append((i, j, mat[i, j]))
                            if anchor_label != other_label:
                                neg_pairs.append((i, j, mat[i, j]))

                correct_a1, correct_p = [], []
                correct_a2, correct_n = [], []
                for a1, p, ap_sim in pos_pairs:
                    most_difficult = (
                        c_f.neg_inf(dtype)
                        if distance.is_inverted
                        else c_f.pos_inf(dtype)
                    )
                    for a2, n, an_sim in neg_pairs:
                        if a2 == a1:
                            condition = (
                                (an_sim > most_difficult)
                                if distance.is_inverted
                                else (an_sim < most_difficult)
                            )
                            if condition:
                                most_difficult = an_sim
                    condition = (
                        (ap_sim < most_difficult + epsilon)
                        if distance.is_inverted
                        else (ap_sim > most_difficult - epsilon)
                    )
                    if condition:
                        correct_a1.append(a1)
                        correct_p.append(p)

                for a2, n, an_sim in neg_pairs:
                    most_difficult = (
                        c_f.pos_inf(dtype)
                        if distance.is_inverted
                        else c_f.neg_inf(dtype)
                    )
                    for a1, p, ap_sim in pos_pairs:
                        if a2 == a1:
                            condition = (
                                (ap_sim < most_difficult)
                                if distance.is_inverted
                                else (ap_sim > most_difficult)
                            )
                            if condition:
                                most_difficult = ap_sim
                    condition = (
                        (an_sim > most_difficult - epsilon)
                        if distance.is_inverted
                        else (an_sim < most_difficult + epsilon)
                    )
                    if condition:
                        correct_a2.append(a2)
                        correct_n.append(n)

                correct_pos_pairs = set([(a, p) for a, p in zip(correct_a1, correct_p)])
                correct_neg_pairs = set([(a, n) for a, n in zip(correct_a2, correct_n)])

                a1, p1, a2, n2 = miner(embeddings, labels)
                pos_pairs = set([(a.item(), p.item()) for a, p in zip(a1, p1)])
                neg_pairs = set([(a.item(), n.item()) for a, n in zip(a2, n2)])

                self.assertTrue(pos_pairs == correct_pos_pairs)
                self.assertTrue(neg_pairs == correct_neg_pairs)

    def test_empty_output(self):
        miner = MultiSimilarityMiner(0.1)
        batch_size = 32
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(batch_size, 64).type(dtype).to(TEST_DEVICE)
            labels = torch.arange(batch_size)
            a1, p, _, _ = miner(embeddings, labels)
            self.assertTrue(len(a1) == 0)
            self.assertTrue(len(p) == 0)
