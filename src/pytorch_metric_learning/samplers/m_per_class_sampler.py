from collections.abc import Iterator

import torch
from torch.utils.data.sampler import Sampler

from .sampling import Labels, as_label_list, choose, indices_by_label, shuffled


class MPerClassSampler(Sampler[int]):
    def __init__(
        self,
        labels: Labels,
        m: int,
        batch_size: int | None = None,
        num_samples: int = 100000,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self.m_per_class = int(m)
        self.batch_size = int(batch_size) if batch_size is not None else batch_size
        if self.m_per_class <= 0:
            raise ValueError("m must be positive")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_samples < 0:
            raise ValueError("num_samples must be nonnegative")
        self.generator = generator
        self.labels_to_indices = indices_by_label(as_label_list(labels))
        self.labels = list(self.labels_to_indices)
        if not self.labels:
            raise ValueError("labels must not be empty")
        self.length_of_single_pass = self.m_per_class * len(self.labels)
        self.list_size = num_samples
        self.adjust_list_size()

    def adjust_list_size(self) -> None:
        if self.batch_size is None:
            if self.length_of_single_pass < self.list_size:
                self.list_size -= (self.list_size) % (self.length_of_single_pass)
        else:
            if self.list_size < self.batch_size:
                raise ValueError("num_samples must be at least batch_size")
            if self.length_of_single_pass < self.batch_size:
                raise ValueError("m * (number of unique labels) must be >= batch_size")
            if self.batch_size % self.m_per_class != 0:
                raise ValueError(
                    "m_per_class must divide batch_size without any remainder"
                )
            self.list_size -= self.list_size % self.batch_size

    def __len__(self) -> int:
        return self.list_size

    def __iter__(self) -> Iterator[int]:
        idx_list = [0] * self.list_size
        i = 0
        for _ in range(self.calculate_num_iters()):
            curr_label_set = shuffled(self.labels, self.generator)
            if self.batch_size is not None:
                curr_label_set = curr_label_set[: self.batch_size // self.m_per_class]
            for label in curr_label_set:
                chosen = choose(
                    self.labels_to_indices[label], self.m_per_class, self.generator
                )
                idx_list[i : i + self.m_per_class] = chosen.tolist()
                i += self.m_per_class
        return iter(idx_list[: self.list_size])

    def calculate_num_iters(self) -> int:
        divisor = (
            self.length_of_single_pass if self.batch_size is None else self.batch_size
        )
        return self.list_size // divisor if divisor < self.list_size else 1
