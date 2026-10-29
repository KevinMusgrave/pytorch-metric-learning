from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from pytorch_metric_learning import miners
from pytorch_metric_learning.utils import indexing


def subset_weights(
    dataset_size: int,
    subset_indices: Tensor,
    subset_labels: Tensor,
    mined_tuples: Sequence[Tensor],
) -> Tensor:
    weights = torch.zeros(dataset_size, device=subset_labels.device)
    weights[subset_indices] = indexing.convert_to_weights(
        mined_tuples, subset_labels, dtype=weights.dtype
    )
    return weights


def main() -> None:
    torch.manual_seed(17)
    data = torch.randn(64, 8)
    labels = torch.arange(8).repeat_interleave(8)
    model = nn.Linear(8, 4)
    subset = torch.randperm(len(data))[:32]
    model.eval()
    with torch.no_grad():
        embeddings = model(data[subset])
    tuples = miners.MultiSimilarityMiner()(embeddings, labels[subset])
    weights = subset_weights(len(data), subset, labels[subset], tuples)
    sampler = WeightedRandomSampler(
        weights.tolist(),
        num_samples=32,
        replacement=True,
        generator=torch.Generator().manual_seed(17),
    )
    loader = DataLoader(TensorDataset(data, labels), batch_size=8, sampler=sampler)
    print(f"sampled batches: {[batch_labels.tolist() for _, batch_labels in loader]}")


if __name__ == "__main__":
    main()
