import unittest
from typing import Any

import torch

from pytorch_metric_learning.samplers import MPerClassSampler


def next_or_restart(iterator, iterable):
    try:
        return iterator, next(iterator)
    except StopIteration:
        iterator = iter(iterable)
        return iterator, next(iterator)


class TestMPerClassSampler(unittest.TestCase):
    def test_mperclass_sampler(self):
        batch_size = 100
        m = 5
        num_samples = 9999
        num_labels = 100
        labels = torch.randint(low=0, high=num_labels, size=(10000,))
        sampler = MPerClassSampler(labels=labels, m=m, num_samples=num_samples)
        self.assertTrue(
            len(sampler) == (m * num_labels) * (num_samples // (m * num_labels))
        )
        iterable = iter(sampler)
        for _ in range(10):
            x = [next(iterable) for _ in range(batch_size)]
            curr_labels = labels[x]
            unique_labels, counts = torch.unique(curr_labels, return_counts=True)
            self.assertTrue(len(unique_labels) == batch_size // m)
            self.assertTrue(torch.all(counts == m))

    def test_mperclass_sampler_with_batch_size(self):
        for batch_size in [4, 50, 99, 100, 1024]:
            for m in [1, 5, 10, 17, 50]:
                for num_labels in [2, 10, 55]:
                    for num_samples in [100, 999, 10000]:
                        self.check_sampler_with_batch_size(
                            batch_size, m, num_labels, num_samples
                        )

    def check_sampler_with_batch_size(self, batch_size, m, num_labels, num_samples):
        fake_embeddings = torch.randn(10000, 2)
        labels = torch.randint(low=0, high=num_labels, size=(10000,))
        dataset = torch.utils.data.TensorDataset(fake_embeddings, labels)
        args: list[Any] = [labels, m, batch_size, num_samples]
        if (
            (num_samples < batch_size)
            or (m * num_labels < batch_size)
            or (batch_size % m != 0)
        ):
            self.assertRaises(ValueError, MPerClassSampler, *args)
            return
        sampler = MPerClassSampler(*args)
        iterator = iter(sampler)
        for _ in range(1000):
            x = []
            for _ in range(batch_size):
                iterator, curr_batch = next_or_restart(iterator, sampler)
                x.append(curr_batch)
            self.assert_m_per_class(labels[x], batch_size, m)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=False,
        )
        for _ in range(2):
            for _, curr_labels in dataloader:
                self.assert_m_per_class(curr_labels, batch_size, m)

    def assert_m_per_class(self, curr_labels, batch_size, m):
        unique_labels, counts = torch.unique(curr_labels, return_counts=True)
        self.assertTrue(len(unique_labels) == batch_size // m)
        self.assertTrue(torch.all(counts == m))


if __name__ == "__main__":
    unittest.main()
