import numpy as np
import torch
from torch.utils.data.sampler import Sampler

from ..utils import common_functions as c_f


class FixedSetOfTriplets(Sampler):
    """
    Upon initialization, this will create num_triplets triplets based on
    the labels provided in labels_to_indices. This is for experimental purposes,
    to see how algorithms perform when the only ground truth is a set of
    triplets, rather than having explicit labels.
    """

    def __init__(self, labels, num_triplets):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.labels_to_indices = c_f.get_labels_to_indices(labels)
        self.num_triplets = int(num_triplets)
        self.create_fixed_set_of_triplets()

    def __len__(self):
        return self.fixed_set_of_triplets.shape[0] * 3

    def __iter__(self):
        c_f.NUMPY_RANDOM.shuffle(self.fixed_set_of_triplets)
        flattened = self.fixed_set_of_triplets.flatten().tolist()
        return iter(flattened)

    def create_fixed_set_of_triplets(self):
        """
        This creates self.fixed_set_of_triplets, which is a numpy array of size
        (num_triplets, 3). Each row is a triplet of indices: (a, p, n), where
        a=anchor, p=positive, and n=negative. Each triplet is created by first
        randomly sampling two classes, then randomly sampling an anchor, positive,
        and negative.
        """
        assert self.num_triplets > 0
        self.fixed_set_of_triplets = np.ones((self.num_triplets, 3), dtype=int) * -1
        label_list = list(self.labels_to_indices.keys())
        for i in range(self.num_triplets):
            anchor_label, negative_label = c_f.NUMPY_RANDOM.choice(
                label_list, size=2, replace=False
            )
            anchor_list = self.labels_to_indices[anchor_label]
            negative_list = self.labels_to_indices[negative_label]
            anchor, positive = c_f.safe_random_choice(anchor_list, size=2)
            negative = c_f.NUMPY_RANDOM.choice(negative_list, replace=False)
            self.fixed_set_of_triplets[i, :] = np.array([anchor, positive, negative])
