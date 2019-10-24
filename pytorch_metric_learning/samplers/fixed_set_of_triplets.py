from torch.utils.data.sampler import Sampler
from ..utils import common_functions as c_f
import numpy as np

class FixedSetOfTriplets(Sampler):
    """
    Upon initialization, this will create num_triplets triplets based on
    the labels provided in labels_to_indices. This is for experimental purposes,
    to see how algorithms perform when the only ground truth is a set of
    triplets, rather than having explicit labels.
    """

    def __init__(self, labels_to_indices, num_triplets, hierarchy_level=0):
        self.create_fixed_set_of_triplets(
            labels_to_indices[hierarchy_level], num_triplets
        )

    def __len__(self):
        return self.fixed_set_of_triplets.shape[0] * 3

    def __iter__(self):
        np.random.shuffle(self.fixed_set_of_triplets)
        flattened = self.fixed_set_of_triplets.flatten().tolist()
        return iter(flattened)

    def create_fixed_set_of_triplets(self, labels_to_indices, num_triplets):
        """
        This creates self.fixed_set_of_triplets, which is a numpy array of size
        (num_triplets, 3). Each row is a triplet of indices: (a, p, n), where
        a=anchor, p=positive, and n=negative. Each triplet is created by first
        randomly sampling two classes, then randomly sampling an anchor, positive,
        and negative.
        """
        num_triplets = int(num_triplets)
        assert num_triplets > 0
        self.fixed_set_of_triplets = np.ones((num_triplets, 3), dtype=np.int) * -1
        label_list = list(labels_to_indices.keys())
        for i in range(num_triplets):
            anchor_label, negative_label = random.sample(label_list, 2)
            anchor_list = labels_to_indices[anchor_label]
            negative_list = labels_to_indices[negative_label]
            anchor, positive = c_f.safe_random_choice(anchor_list, size=2)
            negative = np.random.choice(negative_list, replace=False)
            self.fixed_set_of_triplets[i, :] = np.array([anchor, positive, negative])