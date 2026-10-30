import itertools
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data.sampler import Sampler

from ._sampling import choose, shuffled


class HierarchicalSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: Tensor | np.ndarray | Sequence[Sequence[int]],
        batch_size: int,
        samples_per_class: int | Literal["all"],
        batches_per_super_tuple: int = 4,
        super_classes_per_batch: int = 2,
        inner_label: int = 0,
        outer_label: int = 1,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        labels = torch.as_tensor(labels)
        self.batch_size = batch_size
        self.batches_per_super_tuple = batches_per_super_tuple
        self.samples_per_class = samples_per_class
        self.super_classes_per_batch = super_classes_per_batch
        self.generator = generator

        if batch_size <= 0 or super_classes_per_batch <= 0:
            raise ValueError("batch_size and super_classes_per_batch must be positive")
        if batches_per_super_tuple < 0:
            raise ValueError("batches_per_super_tuple must be nonnegative")

        if self.batch_size % super_classes_per_batch != 0:
            raise ValueError(
                f"batch_size should be a multiple of {super_classes_per_batch}"
            )
        self.sub_batch_len = self.batch_size // super_classes_per_batch

        if self.samples_per_class != "all":
            if self.samples_per_class <= 0:
                raise ValueError('samples_per_class must be positive or "all"')
            if self.sub_batch_len % self.samples_per_class != 0:
                raise ValueError("batch_size not a multiple of samples_per_class")

        groups = defaultdict(lambda: defaultdict(list))
        for idx, (super_label, label) in enumerate(
            zip(
                labels[:, outer_label].tolist(),
                labels[:, inner_label].tolist(),
                strict=True,
            )
        ):
            groups[super_label][label].append(idx)
        self.super_image_lists = {
            super_label: {label: torch.tensor(v) for label, v in classes.items()}
            for super_label, classes in groups.items()
        }
        self.super_pairs = list(
            itertools.combinations(self.super_image_lists, super_classes_per_batch)
        )

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batches()

    def __len__(self) -> int:
        return len(self.super_pairs) * self.batches_per_super_tuple

    def batches(self) -> list[list[int]]:
        batches = []
        for combination in self.super_pairs:
            for _ in range(self.batches_per_super_tuple):
                batch: list[int] = []
                for super_label in combination:
                    batch.extend(self.sub_batch(self.super_image_lists[super_label]))
                batches.append(shuffled(batch, self.generator))
        return shuffled(batches, self.generator)

    def sub_batch(self, classes: dict[int, Tensor]) -> list[int]:
        sub_batch: list[int] = []
        for label in shuffled(list(classes), self.generator):
            if len(sub_batch) >= self.sub_batch_len:
                break
            instances = classes[label]
            size = (
                len(instances)
                if self.samples_per_class == "all"
                else self.samples_per_class
            )
            if len(sub_batch) + size > self.sub_batch_len:
                continue
            sub_batch.extend(choose(instances, size, self.generator).tolist())
        return sub_batch
