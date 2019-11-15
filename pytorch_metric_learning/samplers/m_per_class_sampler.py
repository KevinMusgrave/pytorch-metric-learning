from torch.utils.data.sampler import Sampler
from ..utils import common_functions as c_f
import numpy as np

# modified from
# https://raw.githubusercontent.com/bnulihaixia/Deep_metric/master/utils/sampler.py
class MPerClassSampler(Sampler):
    """
    At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned
    Args:
        labels_to_indices: a dictionary mapping dataset labels to lists of
                            indices that have that label
        m: the number of samples per class to fetch at every iteration. If a
                    class has less than m samples, then there will be duplicates
                    in the returned batch
        hierarchy_level: which level of labels will be used to form each batch.
                        The default is 0, because most use-cases will have
                        1 label per datapoint. But for example, iNat has 7
                        labels per datapoint, in which case hierarchy_level could
                        be set to a number between 0 and 6.
    """

    def __init__(self, labels_to_indices, m, hierarchy_level=0):
        self.m_per_class = int(m)
        self.labels_to_indices = labels_to_indices
        self.set_hierarchy_level(hierarchy_level)

    def __len__(self):
        return len(self.labels) * self.m_per_class

    def __iter__(self):
        ret = []
        for _ in range(1000):
            np.random.RandomState().shuffle(self.labels)
            for label in self.labels:
                t = self.curr_labels_to_indices[label]
                t = c_f.safe_random_choice(t, size=self.m_per_class)
                ret.extend(t)
        return iter(ret)

    def set_hierarchy_level(self, hierarchy_level):
        self.curr_labels_to_indices = self.labels_to_indices[hierarchy_level]
        self.labels = list(self.curr_labels_to_indices.keys())