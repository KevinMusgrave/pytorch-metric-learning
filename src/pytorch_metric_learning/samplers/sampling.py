from collections import defaultdict

import torch


def as_label_list(labels):
    if torch.is_tensor(labels):
        return labels.tolist()
    if hasattr(labels, "tolist"):
        return labels.tolist()
    return list(labels)


def indices_by_label(labels):
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    return {label: torch.tensor(indices) for label, indices in groups.items()}


def shuffled(items, generator):
    order = torch.randperm(len(items), generator=generator).tolist()
    return [items[i] for i in order]


def choose(indices, size, generator):
    if len(indices) >= size:
        return indices[torch.randperm(len(indices), generator=generator)[:size]]
    return indices[torch.randint(len(indices), (size,), generator=generator)]
