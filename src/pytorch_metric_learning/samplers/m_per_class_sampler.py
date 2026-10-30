from collections.abc import Iterator

import torch
from torch.utils.data.sampler import Sampler

from ..types import DatasetLabels
from ._sampling import Rotation, as_label_list, indices_by_label


class MPerClassSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: DatasetLabels,
        m: int,
        batch_size: int,
        num_batches: int | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self.m_per_class = int(m)
        self.batch_size = int(batch_size)
        if self.m_per_class <= 0:
            raise ValueError("m must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.batch_size % self.m_per_class != 0:
            raise ValueError("m must divide batch_size without any remainder")
        self.generator = generator
        self.labels_to_indices = {
            label: indices.tolist()
            for label, indices in indices_by_label(as_label_list(labels)).items()
        }
        self.labels = list(self.labels_to_indices)
        if not self.labels:
            raise ValueError("labels must not be empty")
        self.classes_per_batch = self.batch_size // self.m_per_class
        if self.classes_per_batch > len(self.labels):
            raise ValueError("m * (number of unique labels) must be >= batch_size")
        self.num_batches = (
            self.batches_to_cover_every_example()
            if num_batches is None
            else int(num_batches)
        )
        if self.num_batches <= 0:
            raise ValueError("num_batches must be positive")

    def batches_to_cover_every_example(self) -> int:
        largest_class = max(map(len, self.labels_to_indices.values()))
        passes = -(-largest_class // self.m_per_class)
        return -(-passes * len(self.labels) // self.classes_per_batch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        classes = Rotation(self.labels, self.generator)
        members = {
            label: Rotation(indices, self.generator)
            for label, indices in self.labels_to_indices.items()
        }
        for _ in range(self.num_batches):
            yield [
                index
                for label in classes.take(self.classes_per_batch)
                for index in members[label].take(self.m_per_class)
            ]
