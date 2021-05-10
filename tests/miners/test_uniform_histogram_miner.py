import unittest

import torch

from pytorch_metric_learning.distances import LpDistance, SNRDistance
from pytorch_metric_learning.miners import UniformHistogramMiner
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from .. import TEST_DEVICE, TEST_DTYPES


class TestUniformHistogramMiner(unittest.TestCase):
    def test_uniform_histogram_miner(self):
        torch.manual_seed(93612)
        batch_size = 128
        embedding_size = 32
        num_bins, pos_per_bin, neg_per_bin = 100, 25, 123
        for distance in [
            LpDistance(p=1),
            LpDistance(p=2),
            LpDistance(normalize_embeddings=False),
            SNRDistance(),
        ]:
            miner = UniformHistogramMiner(
                num_bins=num_bins,
                pos_per_bin=pos_per_bin,
                neg_per_bin=neg_per_bin,
                distance=distance,
            )
            for dtype in TEST_DTYPES:
                embeddings = torch.randn(
                    batch_size, embedding_size, device=TEST_DEVICE, dtype=dtype
                )
                labels = torch.randint(0, 2, size=(batch_size,), device=TEST_DEVICE)

                a1, p, a2, n = lmu.get_all_pairs_indices(labels)
                dist_mat = distance(embeddings)
                pos_pairs = dist_mat[a1, p]
                neg_pairs = dist_mat[a2, n]

                a1, p, a2, n = miner(embeddings, labels)

                if dtype == torch.float16:
                    continue  # histc doesn't work for Half tensor

                pos_histogram = torch.histc(
                    dist_mat[a1, p],
                    bins=num_bins,
                    min=torch.min(pos_pairs),
                    max=torch.max(pos_pairs),
                )
                neg_histogram = torch.histc(
                    dist_mat[a2, n],
                    bins=num_bins,
                    min=torch.min(neg_pairs),
                    max=torch.max(neg_pairs),
                )

                self.assertTrue(
                    torch.all((pos_histogram == pos_per_bin) | (pos_histogram == 0))
                )
                self.assertTrue(
                    torch.all((neg_histogram == neg_per_bin) | (neg_histogram == 0))
                )

    def test_no_positives(self):
        miner = UniformHistogramMiner()
        batch_size = 32
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(batch_size, 64).type(dtype).to(TEST_DEVICE)
            labels = torch.arange(batch_size)
            a1, p, _, _ = miner(embeddings, labels)
            self.assertTrue(len(a1) == len(p) == 0)
