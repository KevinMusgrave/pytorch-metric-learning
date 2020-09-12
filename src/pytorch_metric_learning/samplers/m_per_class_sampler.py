import torch
from torch.utils.data.sampler import Sampler
from ..utils import common_functions as c_f

# modified from
# https://raw.githubusercontent.com/bnulihaixia/Deep_metric/master/utils/sampler.py
class MPerClassSampler(Sampler):
    """
    At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned
    """
    def __init__(self, labels, m, batch_size=None, length_before_new_iter=100000):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.m_per_class = int(m)
        self.batch_size = int(batch_size) if batch_size is not None else batch_size
        self.labels_to_indices = c_f.get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        self.length_of_single_pass = self.m_per_class*len(self.labels)
        self.list_size = length_before_new_iter
        if self.batch_size is None:
            if self.length_of_single_pass < self.list_size:
                self.list_size -= (self.list_size) % (self.length_of_single_pass)
        else:
            assert self.list_size >= self.batch_size
            assert self.length_of_single_pass >= self.batch_size, \
                "m * (number of unique labels) must be >= batch_size"
            assert (self.batch_size % self.m_per_class) == 0, \
                "m_per_class must divide batch_size without any remainder"
            self.list_size -= self.list_size % self.batch_size
            
    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = [0]*self.list_size
        i = 0
        num_iters = self.calculate_num_iters()
        for _ in range(num_iters):
            c_f.NUMPY_RANDOM.shuffle(self.labels)
            if self.batch_size is None:
                curr_label_set = self.labels
            else:
                curr_label_set = self.labels[:self.batch_size // self.m_per_class]  
            for label in curr_label_set:
                t = self.labels_to_indices[label]
                idx_list[i:i+self.m_per_class] = c_f.safe_random_choice(t, size=self.m_per_class)
                i += self.m_per_class
        return iter(idx_list)


    def calculate_num_iters(self):
        divisor = self.length_of_single_pass if self.batch_size is None else self.batch_size
        return self.list_size // divisor if divisor < self.list_size else 1