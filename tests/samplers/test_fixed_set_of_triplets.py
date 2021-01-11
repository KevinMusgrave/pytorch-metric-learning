import unittest

import torch

from pytorch_metric_learning.miners import EmbeddingsAlreadyPackagedAsTriplets
from pytorch_metric_learning.samplers import FixedSetOfTriplets
from pytorch_metric_learning.utils import common_functions as c_f


class TestFixedSetOfTriplet(unittest.TestCase):
    def test_fixed_set_of_triplets_with_batch_size(self):
        miner = EmbeddingsAlreadyPackagedAsTriplets()
        for batch_size in [3, 33, 99]:
            batch_of_fake_embeddings = torch.randn(batch_size, 2)
            for num_labels in [2, 10, 55]:
                for num_triplets in [100, 999, 10000]:
                    fake_embeddings = torch.randn(10000, 2)
                    labels = torch.randint(low=0, high=num_labels, size=(10000,))
                    dataset = c_f.EmbeddingDataset(fake_embeddings, labels)
                    sampler = FixedSetOfTriplets(labels, num_triplets)
                    iterator = iter(sampler)
                    for _ in range(1000):
                        x = []
                        for _ in range(batch_size):
                            iterator, curr_batch = c_f.try_next_on_generator(
                                iterator, sampler
                            )
                            x.append(curr_batch)
                        curr_labels = labels[x]
                        a, p, n = miner(batch_of_fake_embeddings, curr_labels)
                        self.assertTrue(len(a) == batch_size // 3)

                    dataloader = torch.utils.data.DataLoader(
                        dataset, batch_size=batch_size, sampler=sampler, drop_last=True
                    )
                    for (embeddings, curr_labels) in dataloader:
                        a, p, n = miner(batch_of_fake_embeddings, curr_labels)
                        self.assertTrue(len(a) == batch_size // 3)


if __name__ == "__main__":
    unittest.main()
