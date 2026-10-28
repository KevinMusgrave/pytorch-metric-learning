import itertools
from collections import defaultdict

import torch
from torch.utils.data.sampler import Sampler

from .sampling import choose, shuffled


class HierarchicalSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels,
        batch_size,
        samples_per_class,
        batches_per_super_tuple=4,
        super_classes_per_batch=2,
        inner_label=0,
        outer_label=1,
        *,
        generator: torch.Generator | None = None,
    ):
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

    def __iter__(self):
        yield from self.batches()

    def __len__(self):
        return len(self.super_pairs) * self.batches_per_super_tuple

    def batches(self):
        batches = []
        for combination in self.super_pairs:
            for _ in range(self.batches_per_super_tuple):
                batch = []
                for super_label in combination:
                    batch.extend(self.sub_batch(self.super_image_lists[super_label]))
                batches.append(shuffled(batch, self.generator))
        return shuffled(batches, self.generator)

    def sub_batch(self, classes):
        sub_batch = []
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
