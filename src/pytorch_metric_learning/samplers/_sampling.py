from collections import defaultdict
from collections.abc import Hashable, Sequence

import numpy as np
import torch
from torch import Tensor

from ..types import DatasetLabels


def as_label_list(labels: DatasetLabels) -> list[Hashable]:
    if isinstance(labels, (Tensor, np.ndarray)):
        return torch.as_tensor(labels).tolist()
    return list(labels)


def indices_by_label(labels: Sequence[Hashable]) -> dict[Hashable, Tensor]:
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    return {label: torch.tensor(indices) for label, indices in groups.items()}


def shuffled[T](items: Sequence[T], generator: torch.Generator | None) -> list[T]:
    order = torch.randperm(len(items), generator=generator).tolist()
    return [items[i] for i in order]


def choose(indices: Tensor, size: int, generator: torch.Generator | None) -> Tensor:
    if len(indices) >= size:
        return indices[torch.randperm(len(indices), generator=generator)[:size]]
    return indices[torch.randint(len(indices), (size,), generator=generator)]
