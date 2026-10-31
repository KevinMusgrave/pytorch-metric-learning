from collections.abc import Sequence
from typing import overload

import torch
from torch import Tensor

from ..types import IndicesTuple, PairIndices, TripletIndices, TripletsPerAnchor
from . import _common

__all__ = [
    "convert_to_pairs",
    "convert_to_triplets",
    "convert_to_weights",
    "get_all_pairs_indices",
    "get_all_triplets_indices",
    "get_random_triplet_indices",
    "pairs_to_triplets",
]


def _get_matches_and_diffs(
    labels: Tensor, ref_labels: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    reference = labels if ref_labels is None else ref_labels
    matches = labels.unsqueeze(1) == reference.unsqueeze(0)
    diffs = ~matches
    if ref_labels is None:
        matches.fill_diagonal_(False)
    return matches, diffs


def get_all_pairs_indices(
    labels: Tensor, ref_labels: Tensor | None = None
) -> PairIndices:
    matches, diffs = _get_matches_and_diffs(labels, ref_labels)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return PairIndices(a1_idx, p_idx, a2_idx, n_idx)


def convert_to_pairs(
    indices_tuple: Sequence[Tensor] | None,
    labels: Tensor | None,
    ref_labels: Tensor | None = None,
) -> PairIndices:
    indices = _common.as_indices_tuple(indices_tuple)
    if indices is None:
        if labels is None:
            raise ValueError("labels are required when indices_tuple is None")
        return get_all_pairs_indices(labels, ref_labels)
    if isinstance(indices, PairIndices):
        return indices
    a, p, n = indices
    return PairIndices(a, p, a, n)


def _convert_to_pos_pairs_with_unique_labels(
    indices_tuple: Sequence[Tensor] | None, labels: Tensor
) -> tuple[Tensor, Tensor]:
    a, p, _, _ = convert_to_pairs(indices_tuple, labels)
    anchor_labels = labels[a]
    unique_labels, inverse = torch.unique(anchor_labels, return_inverse=True)
    first = torch.full_like(unique_labels, len(a), dtype=torch.long).scatter_reduce(
        0, inverse, torch.arange(len(a), device=a.device), reduce="amin"
    )
    return a[first], p[first]


def pairs_to_triplets(pairs: Sequence[Tensor]) -> TripletIndices:
    indices = _common.as_indices_tuple(pairs)
    if not isinstance(indices, PairIndices):
        raise ValueError("pairs_to_triplets requires four pair tensors")
    return _pairs_to_triplets(indices)


def _pairs_to_triplets(pairs: PairIndices) -> TripletIndices:
    a1, p, a2, n = pairs
    sorted_a2, order = torch.sort(a2, stable=True)
    starts = torch.searchsorted(sorted_a2, a1, side="left")
    per_pair = torch.searchsorted(sorted_a2, a1, side="right") - starts
    pair_idx = torch.repeat_interleave(
        torch.arange(len(a1), device=a1.device), per_pair
    )
    pair_offsets = per_pair.cumsum(0) - per_pair
    within = torch.arange(len(pair_idx), device=a1.device) - pair_offsets[pair_idx]
    negatives = n[order[starts[pair_idx] + within]]
    return TripletIndices(a1[pair_idx], p[pair_idx], negatives)


def get_all_triplets_indices(
    labels: Tensor, ref_labels: Tensor | None = None
) -> TripletIndices:
    return _pairs_to_triplets(get_all_pairs_indices(labels, ref_labels))


def get_random_triplet_indices(
    labels: Tensor,
    ref_labels: Tensor | None = None,
    t_per_anchor: int | None = None,
    weights: Tensor | None = None,
) -> TripletIndices:
    matches, diffs = _get_matches_and_diffs(labels, ref_labels)
    return _sample_triplets(matches, diffs, t_per_anchor, weights)


def _sample_triplets(
    pos_mask: Tensor,
    neg_mask: Tensor,
    t_per_anchor: int | None = None,
    weights: Tensor | None = None,
) -> TripletIndices:
    pos_probs = pos_mask.to(torch.float32)
    neg_probs = neg_mask.to(torch.float32)
    if weights is not None:
        neg_probs = neg_probs * weights.to(torch.float32)
    pos_count = pos_mask.sum(dim=1)
    valid = (pos_count > 0) & (neg_probs.sum(dim=1) > 0)
    per_anchor = (
        pos_count if t_per_anchor is None else torch.full_like(pos_count, t_per_anchor)
    )
    per_anchor = per_anchor * valid
    k_max = int(per_anchor.max()) if len(per_anchor) else 0
    if k_max == 0:
        empty = torch.empty(0, dtype=torch.long, device=pos_mask.device)
        return TripletIndices(empty, empty.clone(), empty.clone())
    anchors = torch.where(valid)[0]
    positives = torch.multinomial(pos_probs[anchors], k_max, replacement=True)
    negatives = torch.multinomial(neg_probs[anchors], k_max, replacement=True)
    keep = torch.arange(k_max, device=pos_mask.device) < per_anchor[anchors].unsqueeze(
        1
    )
    anchor_grid = anchors.unsqueeze(1).expand(-1, k_max)
    return TripletIndices(anchor_grid[keep], positives[keep], negatives[keep])


def convert_to_triplets(
    indices_tuple: Sequence[Tensor] | None,
    labels: Tensor | None,
    ref_labels: Tensor | None = None,
    t_per_anchor: TripletsPerAnchor = "all",
) -> TripletIndices:
    indices = _common.as_indices_tuple(indices_tuple)
    if indices is None:
        if labels is None:
            raise ValueError("labels are required when indices_tuple is None")
        if t_per_anchor == "all":
            return get_all_triplets_indices(labels, ref_labels)
        return get_random_triplet_indices(labels, ref_labels, t_per_anchor=t_per_anchor)
    if isinstance(indices, TripletIndices):
        return indices
    return _pairs_to_triplets(indices)


def convert_to_weights(
    indices_tuple: Sequence[Tensor] | None,
    labels: Tensor,
    dtype: torch.dtype,
    using_ref: bool = False,
) -> Tensor:
    tuples = _common.as_indices_tuple(indices_tuple)
    if tuples is None:
        return torch.ones(labels.shape[0], device=labels.device, dtype=dtype)
    indices = _get_anchors(tuples) if using_ref else torch.cat(tuples)
    counts = torch.bincount(indices, minlength=labels.shape[0]).float()
    if len(counts) == 0:
        return counts.to(dtype)
    return (counts / counts.max().clamp_min(1)).to(dtype)


def _get_anchors(indices_tuple: IndicesTuple) -> Tensor:
    if isinstance(indices_tuple, TripletIndices):
        return indices_tuple.anchors
    return torch.cat([indices_tuple.pos_anchors, indices_tuple.neg_anchors], dim=0)


def _concatenate_indices_tuples(it1: IndicesTuple, it2: IndicesTuple) -> IndicesTuple:
    if isinstance(it1, TripletIndices):
        return TripletIndices(*(torch.cat(x) for x in zip(it1, it2, strict=True)))
    return PairIndices(*(torch.cat(x) for x in zip(it1, it2, strict=True)))


@overload
def _remove_self_comparisons(
    indices_tuple: PairIndices, curr_batch_idx: Tensor
) -> PairIndices: ...


@overload
def _remove_self_comparisons(
    indices_tuple: TripletIndices, curr_batch_idx: Tensor
) -> TripletIndices: ...


def _remove_self_comparisons(
    indices_tuple: IndicesTuple, curr_batch_idx: Tensor
) -> IndicesTuple:
    if len(curr_batch_idx) == 0:
        return indices_tuple
    keep = indices_tuple[1] != curr_batch_idx[indices_tuple[0]]
    if isinstance(indices_tuple, TripletIndices):
        a, p, n = indices_tuple
        return TripletIndices(a[keep], p[keep], n[keep])
    a1, p, a2, n = indices_tuple
    return PairIndices(a1[keep], p[keep], a2, n)


def _keep_per_anchor(anchors: Tensor, limit: int) -> Tensor:
    shuffled = torch.randperm(len(anchors), device=anchors.device)
    grouped, order = torch.sort(anchors[shuffled], stable=True)
    first = torch.searchsorted(grouped, grouped, side="left")
    rank = torch.arange(len(anchors), device=anchors.device) - first
    return torch.sort(shuffled[order[rank < limit]]).values


def _limit_per_anchor(indices_tuple: IndicesTuple, limit: int) -> IndicesTuple:
    if isinstance(indices_tuple, TripletIndices):
        keep = _keep_per_anchor(indices_tuple.anchors, limit)
        return TripletIndices(*(x[keep] for x in indices_tuple))
    a1, p, a2, n = indices_tuple
    keep_pos = _keep_per_anchor(a1, limit)
    keep_neg = _keep_per_anchor(a2, limit)
    return PairIndices(a1[keep_pos], p[keep_pos], a2[keep_neg], n[keep_neg])


def _pair_masks(
    pairs: PairIndices, shape: tuple[int, int], device: torch.device
) -> tuple[Tensor, Tensor]:
    a1, p, a2, n = pairs
    pos_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    neg_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    pos_mask[a1, p] = True
    neg_mask[a2, n] = True
    return pos_mask, neg_mask
