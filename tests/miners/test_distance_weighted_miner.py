import unittest

import torch

from pytorch_metric_learning.miners import DistanceWeightedMiner
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from .. import TEST_DEVICE, TEST_DTYPES


class TestDistanceWeightedMiner(unittest.TestCase):
    def test_distance_weighted_miner(self, with_ref_labels=False):
        for dtype in TEST_DTYPES:
            embedding_angles = torch.arange(0, 256)
            embeddings = torch.tensor(
                [c_f.angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            ref_embeddings = embeddings.clone() if with_ref_labels else None
            labels = torch.randint(low=0, high=2, size=(256,))
            ref_labels = (
                torch.randint(low=0, high=2, size=(256,)) if with_ref_labels else None
            )

            a, _, n = lmu.get_all_triplets_indices(labels, ref_labels)
            if with_ref_labels:
                all_an_dist = torch.nn.functional.pairwise_distance(
                    embeddings[a], ref_embeddings[n], 2
                )
            else:
                all_an_dist = torch.nn.functional.pairwise_distance(
                    embeddings[a], embeddings[n], 2
                )
            min_an_dist = torch.min(all_an_dist)

            cutoffs = [0] + list(range(5, 15))
            for non_zero_cutoff_int in cutoffs:
                non_zero_cutoff = (float(non_zero_cutoff_int) / 10.0) - 0.01
                miner = DistanceWeightedMiner(0, non_zero_cutoff)
                a, p, n = miner(embeddings, labels, ref_embeddings, ref_labels)
                if non_zero_cutoff_int == 0:
                    self.assertTrue(len(a) == len(p) == len(n) == 0)
                    continue
                if with_ref_labels:
                    anchors, positives, negatives = (
                        embeddings[a],
                        ref_embeddings[p],
                        ref_embeddings[n],
                    )
                else:
                    anchors, positives, negatives = (
                        embeddings[a],
                        embeddings[p],
                        embeddings[n],
                    )
                an_dist = torch.nn.functional.pairwise_distance(anchors, negatives, 2)
                self.assertTrue(torch.max(an_dist) <= non_zero_cutoff)
                an_dist_var = torch.var(an_dist)
                an_dist_mean = torch.mean(an_dist)
                target_var = (
                    (non_zero_cutoff - min_an_dist) ** 2
                ) / 12  # variance formula for uniform distribution
                target_mean = (non_zero_cutoff - min_an_dist) / 2
                self.assertTrue(torch.abs(an_dist_var - target_var) / target_var < 0.1)
                self.assertTrue(
                    torch.abs(an_dist_mean - target_mean) / target_mean < 0.1
                )

    def test_distance_weighted_miner_with_ref_labels(self):
        self.test_distance_weighted_miner(with_ref_labels=True)

    def test_empty_output(self):
        miner = DistanceWeightedMiner(0.1, 0.5)
        batch_size = 32
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(batch_size, 64).type(dtype).to(TEST_DEVICE)
            labels = torch.arange(batch_size)
            a, p, n = miner(embeddings, labels)
            self.assertTrue(len(a) == 0)
            self.assertTrue(len(p) == 0)
            self.assertTrue(len(n) == 0)
