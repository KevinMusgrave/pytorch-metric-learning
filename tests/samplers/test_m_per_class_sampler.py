import unittest 
import torch
from pytorch_metric_learning.samplers import MPerClassSampler


class TestMPerClassSampler(unittest.TestCase):
    def test_mperclass_sampler(self):
        batch_size = 100
        m = 5
        labels = torch.randint(low=0, high=100, size=(10000,))
        sampler = MPerClassSampler(labels=labels, m=m)
        iterable = iter(sampler)
        for _ in range(10):
            x = [next(iterable) for _ in range(batch_size)]
            curr_labels = labels[x]
            unique_labels, counts = torch.unique(curr_labels, return_counts=True)
            self.assertTrue(len(unique_labels) == batch_size // m)
            self.assertTrue(torch.all(counts==m))



if __name__ == "__main__":
    unittest.main()