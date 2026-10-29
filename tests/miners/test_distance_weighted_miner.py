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

    def test_default_cutoffs_sample_every_anchor(self):
        for dim in (32, 128, 256):
            for dtype in TEST_DTYPES:
                torch.manual_seed(2026)
                embeddings = torch.randn(64, dim, dtype=dtype, device=TEST_DEVICE)
                labels = torch.arange(32, device=TEST_DEVICE).repeat_interleave(2)
                miner = DistanceWeightedMiner()
                mat = miner.distance(embeddings)
                eligible = (labels.unsqueeze(1) != labels.unsqueeze(0)) & (mat < 1.4)
                a, p, n = miner(embeddings, labels)
                self.assertEqual(len(a), int((eligible.sum(dim=1) > 0).sum()))
                self.assertTrue(torch.equal(labels[a], labels[p]))
                self.assertTrue(torch.all(labels[a] != labels[n]))
                self.assertTrue(torch.all(mat[a, n] < 1.4))

    def test_rows_normalize_independently(self):
        for dtype in TEST_DTYPES:
            basis = torch.eye(256, dtype=dtype, device=TEST_DEVICE)
            near = torch.nn.functional.normalize(basis[0] + 0.5 * basis[1], dim=0)
            far = torch.nn.functional.normalize(basis[2] + 1.2 * basis[3], dim=0)
            embeddings = torch.stack(
                [basis[0], basis[0], near, basis[2], basis[2], far]
            )
            labels = torch.tensor([0, 0, 1, 2, 2, 3], device=TEST_DEVICE)
            a, _, n = DistanceWeightedMiner()(embeddings, labels)
            self.assertEqual(sorted(a.tolist()), [0, 1, 3, 4])
            self.assertTrue(torch.all(n[labels[a] == 0] == 2))
            self.assertTrue(torch.all(n[labels[a] == 2] == 5))
