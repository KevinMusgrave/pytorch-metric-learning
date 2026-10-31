from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor

from ..types import ArrayLike

__all__ = [
    "MatchCounts",
    "ThresholdSelection",
    "match_accuracy",
    "match_balanced_accuracy",
    "match_f1",
    "select_threshold",
]


class MatchCounts(NamedTuple):
    true_positives: Tensor
    false_positives: Tensor
    true_negatives: Tensor
    false_negatives: Tensor


class ThresholdSelection(NamedTuple):
    threshold: float
    value: float


def match_accuracy(counts: MatchCounts) -> Tensor:
    correct = counts.true_positives + counts.true_negatives
    return correct / (correct + counts.false_positives + counts.false_negatives)


def match_balanced_accuracy(counts: MatchCounts) -> Tensor:
    recall = counts.true_positives / (counts.true_positives + counts.false_negatives)
    specificity = counts.true_negatives / (
        counts.true_negatives + counts.false_positives
    )
    return (recall + specificity) / 2


def match_f1(counts: MatchCounts) -> Tensor:
    doubled = 2 * counts.true_positives
    return doubled / (doubled + counts.false_positives + counts.false_negatives)


def select_threshold(
    scores: ArrayLike,
    labels: ArrayLike,
    metric: Callable[[MatchCounts], Tensor],
    *,
    is_inverted: bool,
) -> ThresholdSelection:
    scores_t = torch.as_tensor(scores).detach().cpu()
    labels_t = torch.as_tensor(labels).detach().cpu()
    _check_inputs(scores_t, labels_t)
    order = torch.argsort(scores_t, descending=is_inverted)
    ordered = scores_t[order]
    _, group_sizes = torch.unique_consecutive(ordered, return_counts=True)
    lengths = torch.cat([group_sizes.new_zeros(1), group_sizes.cumsum(0)])
    positives = labels_t[order].cumsum(0, dtype=torch.float64)
    true_positives = torch.cat([positives.new_zeros(1), positives])[lengths]
    false_positives = lengths - true_positives
    false_negatives = positives[-1] - true_positives
    true_negatives = len(ordered) - positives[-1] - false_positives
    values = metric(
        MatchCounts(true_positives, false_positives, true_negatives, false_negatives)
    )
    if not isinstance(values, Tensor) or values.shape != lengths.shape:
        raise ValueError(
            "metric must return a tensor with one value per candidate threshold, "
            f"of shape {tuple(lengths.shape)}"
        )
    if torch.isnan(values).all():
        raise ValueError("metric is undefined for every candidate threshold")
    best = int(torch.where(values.isnan(), -torch.inf, values).argmax())
    threshold = _boundary(ordered, int(lengths[best]), is_inverted)
    return ThresholdSelection(threshold, values[best].item())


def _check_inputs(scores: Tensor, labels: Tensor) -> None:
    if scores.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("scores and labels must be one-dimensional and equal length")
    if not scores.is_floating_point() or labels.dtype != torch.bool:
        raise ValueError("scores must be floating point and labels must be boolean")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if labels.all() or not labels.any():
        raise ValueError("labels must include matching and non-matching pairs")


def _boundary(ordered: Tensor, length: int, is_inverted: bool) -> float:
    if length == 0:
        return torch.inf if is_inverted else -torch.inf
    kept = ordered[length - 1]
    if length == len(ordered):
        return kept.item()
    excluded = ordered[length]
    midpoint = kept / 2 + excluded / 2
    return (kept if midpoint == excluded else midpoint).item()
