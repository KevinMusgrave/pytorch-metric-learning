import unittest

import torch

from pytorch_metric_learning.miners import EmbeddingsAlreadyPackagedAsTriplets
from pytorch_metric_learning.samplers import FixedSetOfTriplets


def next_or_restart(iterator, iterable):
    try:
        return iterator, next(iterator)
    except StopIteration:
        iterator = iter(iterable)
        return iterator, next(iterator)


class TestFixedSetOfTriplet(unittest.TestCase):
    def test_fixed_set_of_triplets_with_batch_size(self):
        miner = EmbeddingsAlreadyPackagedAsTriplets()
        for batch_size in [3, 33, 99]:
            batch_of_fake_embeddings = torch.randn(batch_size, 2)
            for num_labels in [2, 10, 55]:
                for num_triplets in [100, 999, 10000]:
                    fake_embeddings = torch.randn(10000, 2)
                    labels = torch.randint(low=0, high=num_labels, size=(10000,))
                    dataset = torch.utils.data.TensorDataset(fake_embeddings, labels)
                    sampler = FixedSetOfTriplets(labels, num_triplets)
                    iterator = iter(sampler)
                    for _ in range(1000):
                        x = []
                        for _ in range(batch_size):
                            iterator, curr_batch = next_or_restart(iterator, sampler)
                            x.append(curr_batch)
                        curr_labels = labels[x]
                        a, p, n = miner(batch_of_fake_embeddings, curr_labels)
                        self.assertTrue(len(a) == batch_size // 3)
                        self.assertTrue(torch.all(curr_labels[a] == curr_labels[p]))
                        self.assertTrue(torch.all(curr_labels[a] != curr_labels[n]))

                    dataloader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size, sampler=sampler, drop_last=True
                    )
                    for _ in range(2):
                        for _, curr_labels in dataloader:
                            a, p, n = miner(batch_of_fake_embeddings, curr_labels)
                            self.assertTrue(len(a) == batch_size // 3)
                            self.assertTrue(torch.all(curr_labels[a] == curr_labels[p]))
                            self.assertTrue(torch.all(curr_labels[a] != curr_labels[n]))


if __name__ == "__main__":
    unittest.main()
