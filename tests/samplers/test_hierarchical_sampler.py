import unittest
import math
from collections import Counter

import torch

from pytorch_metric_learning.samplers import HierarchicalSampler


class TestHierarchicalSampler(unittest.TestCase):
    def test_mperclass_sampler(self):
        batch_size = 100
        samples_per_class = 5
        batches_per_super_pair = 10
        num_superlabels = 12
        num_labels_per_super_labels = 10
        num_labels = num_superlabels * num_labels_per_super_labels
        labels = torch.randint(low=0, high=num_labels, size=(15000,))
        super_labels = torch.zeros_like(labels)
        for slb in range(num_superlabels):
            super_labels[(labels >= slb * num_labels_per_super_labels) & (labels < (slb + 1) * num_labels_per_super_labels)] = slb
        labels = torch.stack((labels, super_labels), dim=1).numpy()
        sampler = HierarchicalSampler(
            labels=labels,
            batch_size=batch_size,
            samples_per_class=samples_per_class,
            batches_per_super_pair=batches_per_super_pair,
        )
        self.assertTrue(len(sampler) == batches_per_super_pair * math.comb(num_superlabels, 2))
        for j, batch in enumerate(sampler):
            batch_labels = Counter([labels[x, 0] for x in batch])
            batch_super_labels = Counter([labels[x, 1] for x in batch])
            self.assertTrue(len(batch) == batch_size)
            self.assertTrue(len((batch_super_labels)) == 2)
            self.assertTrue(len(set(batch_super_labels.values())) == 1)  # Each super labels has the same number of instances
            self.assertTrue(len(set(batch_labels.values())) == 1)  # Each labels has the samples_per_class number of instances
            self.assertTrue(list(set(batch_labels.values()))[0] == samples_per_class)

            if j == 10:
                break


if __name__ == "__main__":
    unittest.main()
