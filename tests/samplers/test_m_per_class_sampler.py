import unittest 
import torch
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_metric_learning.utils import common_functions as c_f


class TestMPerClassSampler(unittest.TestCase):
    def test_mperclass_sampler(self):
        batch_size = 100
        m = 5
        length_before_new_iter = 9999
        num_labels = 100
        labels = torch.randint(low=0, high=num_labels, size=(10000,))
        sampler = MPerClassSampler(labels=labels, m=m, length_before_new_iter=length_before_new_iter)
        self.assertTrue(len(sampler) == (m*num_labels)*(length_before_new_iter // (m*num_labels)))
        iterable = iter(sampler)
        for _ in range(10):
            x = [next(iterable) for _ in range(batch_size)]
            curr_labels = labels[x]
            unique_labels, counts = torch.unique(curr_labels, return_counts=True)
            self.assertTrue(len(unique_labels) == batch_size // m)
            self.assertTrue(torch.all(counts==m))


    def test_mperclass_sampler_with_batch_size(self):
        for batch_size in [4, 50, 99, 100, 1024]:
            for m in [1, 5, 10, 17, 50]:
                for num_labels in [2, 10, 55]:
                    for length_before_new_iter in [100, 999, 10000]:
                        labels = torch.randint(low=0, high=num_labels, size=(10000,))
                        args = [labels, m, batch_size, length_before_new_iter]
                        if (length_before_new_iter < batch_size) or \
                            (m*num_labels < batch_size) or \
                            (batch_size % m != 0):
                            self.assertRaises(AssertionError, MPerClassSampler, *args)
                            continue
                        else:
                            sampler = MPerClassSampler(*args)
                        iterator = iter(sampler)
                        for _ in range(1000):
                            x = []
                            for _ in range(batch_size):
                                iterator, curr_batch = c_f.try_next_on_generator(iterator, sampler)
                                x.append(curr_batch)
                            curr_labels = labels[x]
                            unique_labels, counts = torch.unique(curr_labels, return_counts=True)
                            self.assertTrue(len(unique_labels) == batch_size // m)
                            self.assertTrue(torch.all(counts==m))




if __name__ == "__main__":
    unittest.main()