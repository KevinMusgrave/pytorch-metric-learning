from collections import defaultdict

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


@torch.inference_mode()
def extract_embeddings(model: nn.Module, loader: DataLoader) -> tuple[Tensor, Tensor]:
    embeddings, labels = [], []
    for data, batch_labels in loader:
        embeddings.append(model(data))
        labels.append(batch_labels)
    return nn.functional.normalize(torch.cat(embeddings), dim=1), torch.cat(labels)


def split_accuracy(
    calculator: AccuracyCalculator,
    query: tuple[Tensor, Tensor],
    references: list[tuple[Tensor, Tensor]],
    query_in_reference: bool,
) -> dict[str, float]:
    if query_in_reference:
        references = [query, *references]
    reference = torch.cat([embeddings for embeddings, _ in references])
    reference_labels = torch.cat([labels for _, labels in references])
    return calculator.get_accuracy(
        query[0], query[1], reference, reference_labels, query_in_reference
    )


def two_stream_accuracy(
    calculator: AccuracyCalculator,
    anchors: Tensor,
    references: Tensor,
    labels: Tensor,
) -> dict[str, float]:
    return calculator.get_accuracy(anchors, labels, references, labels, False)


def parent_conditioned_accuracy(
    calculator: AccuracyCalculator,
    query: Tensor,
    query_labels: Tensor,
    reference: Tensor,
    reference_labels: Tensor,
    query_in_reference: bool,
) -> dict[str, float]:
    results = {}
    levels = range(query_labels.shape[1] - 1)
    for level in levels:
        by_metric = defaultdict(list)
        for parent in torch.unique(query_labels[:, level + 1]):
            query_mask = query_labels[:, level + 1] == parent
            reference_mask = reference_labels[:, level + 1] == parent
            accuracies = calculator.get_accuracy(
                query[query_mask],
                query_labels[query_mask, level],
                reference[reference_mask],
                reference_labels[reference_mask, level],
                query_in_reference,
            )
            for metric, value in accuracies.items():
                by_metric[metric].append(value)
        for metric, values in by_metric.items():
            results[f"{metric}_level{level}"] = sum(values) / len(values)
    if len(levels) > 1:
        for metric in calculator.get_curr_metrics():
            per_level = [results[f"{metric}_level{level}"] for level in levels]
            results[f"AVERAGE_{metric}"] = sum(per_level) / len(per_level)
    return results


def main() -> None:
    torch.manual_seed(31)
    centers = torch.tensor([[4.0, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0]])
    labels = torch.arange(3).repeat(8)
    data = centers[labels] + torch.randn(24, 4)
    model = nn.Linear(4, 3)
    model.eval()
    loader = DataLoader(TensorDataset(data, labels), batch_size=5)
    embeddings, embedding_labels = extract_embeddings(model, loader)
    calculator = AccuracyCalculator(include=("precision_at_1", "r_precision"))
    query = (embeddings[:12], embedding_labels[:12])
    gallery = (embeddings[12:], embedding_labels[12:])
    within = split_accuracy(calculator, query, [], True)
    against_gallery = split_accuracy(calculator, query, [gallery], False)
    print(f"query against itself: {within}")
    print(f"query against gallery: {against_gallery}")


if __name__ == "__main__":
    main()
