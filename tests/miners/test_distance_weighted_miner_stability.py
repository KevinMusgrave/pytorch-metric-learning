import unittest
from unittest.mock import patch

import torch

from pytorch_metric_learning.miners import DistanceWeightedMiner
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class TestDistanceWeightedMinerStability(unittest.TestCase):
    @staticmethod
    def embeddings(dimension, dtype):
        embeddings = torch.zeros(6, dimension, dtype=dtype)
        embeddings[:2, 0] = 1
        embeddings[2:4, 0] = 0.82
        embeddings[2:4, 1] = (1 - 0.82**2) ** 0.5
        embeddings[4:, 0] = 0.2
        embeddings[4:, 1] = (0.2 - 0.82 * 0.2) / (1 - 0.82**2) ** 0.5
        embeddings[4:, 2] = (1 - embeddings[4, 0] ** 2 - embeddings[4, 1] ** 2) ** 0.5
        return embeddings

    def test_per_anchor_distribution(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        sample_triplets = lmu.get_random_triplet_indices
        for dtype in (torch.float16, torch.float32, torch.float64):
            for dimension in (128, 512, 2048):
                with self.subTest(dtype=dtype, dimension=dimension):
                    embeddings = self.embeddings(dimension, dtype)
                    miner = DistanceWeightedMiner()
                    # Compute the intended per-anchor distribution independently
                    # in double precision, using softmax instead of rescaling.
                    distances = miner.distance(embeddings, embeddings).double()
                    distances = distances.clamp(min=miner.cutoff)
                    log_density = (2 - dimension) * distances.log() - (
                        (dimension - 3) / 2
                    ) * torch.log1p(-0.25 * distances.square())
                    excluded = (labels[:, None] == labels[None, :]) | (
                        distances >= miner.nonzero_loss_cutoff
                    )
                    reference = torch.softmax(
                        log_density.masked_fill(excluded, -torch.inf), dim=1
                    )
                    observed = []

                    def sample(*args, **kwargs):
                        observed.append(kwargs["weights"].clone())
                        return sample_triplets(*args, **kwargs)

                    with patch.object(lmu, "get_random_triplet_indices", sample):
                        anchors, positives, negatives = miner(embeddings, labels)
                    self.assertEqual(set(anchors.tolist()), set(range(6)))
                    self.assertTrue(torch.all(labels[anchors] == labels[positives]))
                    self.assertTrue(torch.all(labels[anchors] != labels[negatives]))
                    tolerance = 0.03 if dtype == torch.float16 else 1e-4
                    torch.testing.assert_close(
                        observed[0].double(), reference, atol=tolerance, rtol=tolerance
                    )

    def test_no_eligible_negatives(self):
        embeddings = self.embeddings(512, torch.float32)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        boundary = DistanceWeightedMiner().distance(embeddings, embeddings)[0, 2].item()
        for cutoff in (0.5, boundary):
            miner = DistanceWeightedMiner(nonzero_loss_cutoff=cutoff)
            anchors, positives, negatives = miner(embeddings, labels)
            self.assertEqual(len(anchors), 0)
            self.assertEqual(len(positives), 0)
            self.assertEqual(len(negatives), 0)
        anchors, _, _ = DistanceWeightedMiner()(embeddings, torch.zeros(6))
        self.assertEqual(len(anchors), 0)

    def test_excluded_negatives_do_not_affect_distribution(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        sample_triplets = lmu.get_random_triplet_indices
        for dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                embeddings = torch.zeros(6, 512, dtype=dtype)
                embeddings[:2, 0] = 1
                embeddings[2:4, 0] = 0.2
                embeddings[2:4, 1] = (1 - 0.2**2) ** 0.5
                embeddings[4:, 0] = -0.9995
                embeddings[4:, 2] = (1 - 0.9995**2) ** 0.5
                miner = DistanceWeightedMiner()
                distances = miner.distance(embeddings, embeddings).double()
                distances = distances.clamp(min=miner.cutoff)
                log_density = (2 - 512) * distances.log() - (
                    (512 - 3) / 2
                ) * torch.log1p(-0.25 * distances.square())
                eligible = (
                    (labels[:, None] != labels[None, :])
                    & (distances < miner.nonzero_loss_cutoff)
                    & torch.isfinite(log_density)
                )
                reference = torch.zeros_like(log_density)
                for row in range(len(labels)):
                    if eligible[row].any():
                        reference[row, eligible[row]] = torch.softmax(
                            log_density[row, eligible[row]], dim=0
                        )
                observed = []

                def sample(*args, **kwargs):
                    observed.append(kwargs["weights"].clone())
                    return sample_triplets(*args, **kwargs)

                with patch.object(lmu, "get_random_triplet_indices", sample):
                    anchors, positives, negatives = miner(embeddings, labels)
                self.assertEqual(set(anchors.tolist()), {0, 1, 2, 3})
                self.assertTrue(torch.all(labels[anchors] == labels[positives]))
                self.assertTrue(torch.all(labels[anchors] != labels[negatives]))
                tolerance = 0.03 if dtype == torch.float16 else 1e-4
                torch.testing.assert_close(
                    observed[0].double(), reference, atol=tolerance, rtol=tolerance
                )
