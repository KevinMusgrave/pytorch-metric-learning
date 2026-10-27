import unittest

import torch

from pytorch_metric_learning.miners import MultiSimilarityMiner
from pytorch_metric_learning.samplers import TuplesToWeightsSampler

from .. import TEST_DEVICE


class TestTuplesToWeightsSampler(unittest.TestCase):
    def test_tuplestoweights_sampler(self):
        generator = torch.Generator().manual_seed(31)
        embeddings = torch.randn(64, 8, generator=generator)
        labels = torch.arange(8).repeat_interleave(8)
        dataset = torch.utils.data.TensorDataset(embeddings, labels)
        for subset_size in (None, 32):
            with self.subTest(subset_size=subset_size):
                sampler = TuplesToWeightsSampler(
                    torch.nn.Identity().to(TEST_DEVICE),
                    MultiSimilarityMiner(epsilon=0.1),
                    dataset,
                    subset_size=subset_size,
                    data_device=TEST_DEVICE,
                    dataloader_num_workers=0,
                )
                sampled = torch.tensor(list(sampler), device=TEST_DEVICE)
                weights = sampler.weights
                assert isinstance(weights, torch.Tensor)
                self.assertEqual(len(sampled), subset_size or len(dataset))
                self.assertTrue(torch.all((sampled >= 0) & (sampled < len(dataset))))
                self.assertTrue(torch.all(weights[sampled] > 0))
                self.assertTrue(torch.all(torch.isfinite(weights)))
                if subset_size is not None:
                    self.assertLessEqual(torch.count_nonzero(weights), subset_size)
                else:
                    self.assertEqual(weights.shape, (len(dataset),))
