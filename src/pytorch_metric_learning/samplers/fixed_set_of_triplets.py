import torch
from torch.utils.data.sampler import Sampler

from .sampling import as_label_list, choose, indices_by_label


class FixedSetOfTriplets(Sampler[int]):
    def __init__(
        self, labels, num_triplets, *, generator: torch.Generator | None = None
    ):
        self.generator = generator
        self.labels_to_indices = indices_by_label(as_label_list(labels))
        self.num_triplets = int(num_triplets)
        self.create_fixed_set_of_triplets()

    def __len__(self):
        return self.fixed_set_of_triplets.shape[0] * 3

    def __iter__(self):
        order = torch.randperm(
            len(self.fixed_set_of_triplets), generator=self.generator
        )
        return iter(self.fixed_set_of_triplets[order].flatten().tolist())

    def create_fixed_set_of_triplets(self):
        assert self.num_triplets > 0
        label_list = list(self.labels_to_indices)
        triplets = torch.empty(self.num_triplets, 3, dtype=torch.long)
        for i in range(self.num_triplets):
            classes = torch.randperm(len(label_list), generator=self.generator)[:2]
            anchor_list = self.labels_to_indices[label_list[classes[0]]]
            negative_list = self.labels_to_indices[label_list[classes[1]]]
            triplets[i, :2] = choose(anchor_list, 2, self.generator)
            triplets[i, 2] = choose(negative_list, 1, self.generator)[0]
        self.fixed_set_of_triplets = triplets
