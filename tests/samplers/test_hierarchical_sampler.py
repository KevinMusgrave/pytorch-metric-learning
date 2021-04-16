import math
import unittest
from collections import Counter

import torch

from pytorch_metric_learning.samplers import HierarchicalSampler


class TestHierarchicalSampler(unittest.TestCase):
    def test_hierarchical_sampler(self):
        dataset_size = 15023
        batches_per_super_tuple = 9
        num_superlabels = 12

        for super_classes_per_batch in range(1, 4):
            batch_size = 50 * super_classes_per_batch
            for samples_per_class, num_labels_per_super_labels in [
                (5, 17),
                ("all", 93),
            ]:
                num_labels = num_superlabels * num_labels_per_super_labels
                labels = torch.randint(low=0, high=num_labels, size=(dataset_size,))
                super_labels = torch.zeros_like(labels)
                for slb in range(num_superlabels):
                    super_labels[
                        (labels >= slb * num_labels_per_super_labels)
                        & (labels < (slb + 1) * num_labels_per_super_labels)
                    ] = slb
                labels = torch.stack((labels, super_labels), dim=1).numpy()
                sampler = HierarchicalSampler(
                    labels=labels,
                    batch_size=batch_size,
                    samples_per_class=samples_per_class,
                    batches_per_super_tuple=batches_per_super_tuple,
                    super_classes_per_batch=super_classes_per_batch,
                )
                self.assertTrue(
                    len(sampler)
                    == batches_per_super_tuple
                    * math.comb(num_superlabels, super_classes_per_batch)
                )
                all_batch_super_labels = []
                for j, batch in enumerate(sampler):
                    batch_labels = Counter([labels[x, 0] for x in batch])
                    batch_super_labels = Counter([labels[x, 1] for x in batch])
                    all_batch_super_labels.append(
                        tuple(sorted(batch_super_labels.keys()))
                    )
                    if samples_per_class != "all":
                        self.assertTrue(len(batch) == batch_size)
                        self.assertTrue(
                            all(v == samples_per_class for v in batch_labels.values())
                        )  # Each labels has the samples_per_class number of instances
                        self.assertTrue(
                            all(
                                v == batch_size // super_classes_per_batch
                                for v in batch_super_labels.values()
                            )
                        )  # Each super labels has the same number of instances

                    self.assertTrue(len(batch_super_labels) == super_classes_per_batch)

                self.assertTrue(
                    all(
                        v == batches_per_super_tuple
                        for v in Counter(all_batch_super_labels).values()
                    )
                )


if __name__ == "__main__":
    unittest.main()
