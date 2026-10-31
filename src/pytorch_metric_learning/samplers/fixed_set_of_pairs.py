import math
from collections.abc import Iterator

import torch
from torch import Tensor
from torch.utils.data.sampler import Sampler

from ..types import DatasetLabels
from ._sampling import as_label_list, choose, indices_by_label


class FixedSetOfPairs(Sampler[list[int]]):
    def __init__(
        self,
        labels: DatasetLabels,
        num_pairs: int,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        if num_pairs <= 0:
            raise ValueError("num_pairs must be positive")
        if batch_size <= 0 or batch_size % 2 != 0:
            raise ValueError("batch_size must be a positive even number")
        self.generator = generator
        self.pairs_per_batch = batch_size // 2
        self.labels_to_indices = indices_by_label(as_label_list(labels))
        self.fixed_set_of_pairs = self.create_fixed_set_of_pairs(num_pairs)

    def __len__(self) -> int:
        return math.ceil(len(self.fixed_set_of_pairs) / self.pairs_per_batch)

    def __iter__(self) -> Iterator[list[int]]:
        order = torch.randperm(len(self.fixed_set_of_pairs), generator=self.generator)
        for batch in self.fixed_set_of_pairs[order].split(self.pairs_per_batch):
            yield batch.flatten().tolist()

    def create_fixed_set_of_pairs(self, num_pairs: int) -> Tensor:
        groups = list(self.labels_to_indices.values())
        repeated = [group for group in groups if len(group) >= 2]
        if len(groups) < 2:
            raise ValueError("Pair sampling requires at least two classes")
        if not repeated:
            raise ValueError("Positive pairs require a class with two examples")
        pairs = torch.empty(num_pairs, 2, dtype=torch.long)
        num_positive = (num_pairs + 1) // 2
        for i in range(num_positive):
            group = int(torch.randint(len(repeated), (1,), generator=self.generator))
            pairs[i] = choose(repeated[group], 2, self.generator)
        for i in range(num_positive, num_pairs):
            first, second = torch.randperm(len(groups), generator=self.generator)[:2]
            pairs[i, 0] = choose(groups[first], 1, self.generator)[0]
            pairs[i, 1] = choose(groups[second], 1, self.generator)[0]
        return pairs
