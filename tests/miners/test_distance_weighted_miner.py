import unittest

import torch

from pytorch_metric_learning.miners import DistanceWeightedMiner
from pytorch_metric_learning.utils import indexing

from .. import TEST_DEVICE, TEST_DTYPES
from ..helpers import angle_to_coord


def normalized_planar_embeddings(num_points, dtype):
    return torch.tensor(
        [angle_to_coord(a, normalized=True) for a in torch.arange(0, num_points)],
        requires_grad=True,
        dtype=dtype,
    ).to(TEST_DEVICE)


def uniform_distribution_mean(low, high):
    return (high + low) / 2


def uniform_distribution_variance(low, high):
    return ((high - low) ** 2) / 12


class TestDistanceWeightedMiner(unittest.TestCase):
    def test_distance_weighted_miner(self):
        self.check_distance_weighted_miner(with_ref_labels=False)

    def test_distance_weighted_miner_with_ref_labels(self):
        self.check_distance_weighted_miner(with_ref_labels=True)

    def check_distance_weighted_miner(self, with_ref_labels):
        for dtype in TEST_DTYPES:
            embeddings = normalized_planar_embeddings(256, dtype)
            ref_embeddings = embeddings.clone() if with_ref_labels else None
            labels = torch.randint(low=0, high=2, size=(256,))
            ref_labels = (
                torch.randint(low=0, high=2, size=(256,)) if with_ref_labels else None
            )
            pair_embeddings = embeddings if ref_embeddings is None else ref_embeddings

            a, _, n = indexing.get_all_triplets_indices(labels, ref_labels)
            all_an_dist = torch.nn.functional.pairwise_distance(
                embeddings[a], pair_embeddings[n], 2
            )
            max_an_dist = torch.max(all_an_dist)
            min_an_dist = torch.min(all_an_dist)

            for non_zero_cutoff_int in [0, *range(5, 15)]:
                non_zero_cutoff = (float(non_zero_cutoff_int) / 10.0) - 0.01
                miner = DistanceWeightedMiner(0, non_zero_cutoff)
                a, p, n = miner(embeddings, labels, ref_embeddings, ref_labels)
                if non_zero_cutoff_int == 0:
                    self.assertTrue(len(a) == len(p) == len(n) == 0)
                    continue
                an_dist = torch.nn.functional.pairwise_distance(
                    embeddings[a], pair_embeddings[n], 2
                )
                self.assertTrue(torch.max(an_dist) <= non_zero_cutoff)
                an_dist_var = torch.var(an_dist)
                an_dist_mean = torch.mean(an_dist)
                upper_bound = min(max_an_dist, non_zero_cutoff)
                target_var = uniform_distribution_variance(min_an_dist, upper_bound)
                target_mean = uniform_distribution_mean(min_an_dist, upper_bound)
                self.assertTrue(torch.abs(an_dist_var - target_var) / target_var < 0.1)
                self.assertTrue(
                    torch.abs(an_dist_mean - target_mean) / target_mean < 0.1
                )

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
