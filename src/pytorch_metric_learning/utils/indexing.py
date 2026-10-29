from typing import cast

import torch
from torch import Tensor

from ..types import IndicesTuple, PairIndices, TripletIndices, TripletsPerAnchor

__all__ = [
    "convert_to_pairs",
    "convert_to_triplets",
    "convert_to_weights",
    "get_all_pairs_indices",
    "get_all_triplets_indices",
    "get_random_triplet_indices",
]


def get_matches_and_diffs(
    labels: Tensor, ref_labels: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    if ref_labels is None:
        ref_labels = labels
    matches = labels.unsqueeze(1) == ref_labels.unsqueeze(0)
    diffs = ~matches
    if ref_labels is labels:
        matches.fill_diagonal_(False)
    return matches, diffs


def get_all_pairs_indices(
    labels: Tensor, ref_labels: Tensor | None = None
) -> PairIndices:
    matches, diffs = get_matches_and_diffs(labels, ref_labels)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx


def convert_to_pairs(
    indices_tuple: IndicesTuple | None,
    labels: Tensor | None,
    ref_labels: Tensor | None = None,
) -> PairIndices:
    if indices_tuple is None:
        if labels is None:
            raise ValueError("labels are required when indices_tuple is None")
        return get_all_pairs_indices(labels, ref_labels)
    if len(indices_tuple) == 4:
        return indices_tuple
    a, p, n = indices_tuple
    return a, p, a, n


def convert_to_pos_pairs_with_unique_labels(
    indices_tuple: IndicesTuple | None, labels: Tensor
) -> tuple[Tensor, Tensor]:
    a, p, _, _ = convert_to_pairs(indices_tuple, labels)
    anchor_labels = labels[a]
    unique_labels, inverse = torch.unique(anchor_labels, return_inverse=True)
    first = torch.full_like(unique_labels, len(a), dtype=torch.long).scatter_reduce(
        0, inverse, torch.arange(len(a), device=a.device), reduce="amin"
    )
    return a[first], p[first]


def get_all_triplets_indices(
    labels: Tensor, ref_labels: Tensor | None = None
) -> TripletIndices:
    all_matches, all_diffs = get_matches_and_diffs(labels, ref_labels)
    anchors, positives = torch.where(all_matches)
    negatives = torch.where(all_diffs)[1]
    negative_counts = all_diffs.sum(dim=1)
    triplets_per_pair = negative_counts[anchors]
    pair_indices = torch.repeat_interleave(
        torch.arange(len(anchors), device=labels.device), triplets_per_pair
    )
    anchor_indices = anchors[pair_indices]
    pair_starts = triplets_per_pair.cumsum(0) - triplets_per_pair
    negative_starts = negative_counts.cumsum(0) - negative_counts
    negative_indices = (
        negative_starts[anchor_indices]
        + torch.arange(len(pair_indices), device=labels.device)
        - pair_starts[pair_indices]
    )
    return anchor_indices, positives[pair_indices], negatives[negative_indices]


def get_random_triplet_indices(
    labels: Tensor,
    ref_labels: Tensor | None = None,
    t_per_anchor: int | None = None,
    weights: Tensor | None = None,
) -> TripletIndices:
    matches, diffs = get_matches_and_diffs(labels, ref_labels)
    pos_probs = matches.to(torch.float32)
    neg_probs = diffs.to(torch.float32)
    if weights is not None:
        neg_probs = neg_probs * weights.to(torch.float32)
    pos_count = matches.sum(dim=1)
    valid = (pos_count > 0) & (neg_probs.sum(dim=1) > 0)
    per_anchor = (
        pos_count if t_per_anchor is None else torch.full_like(pos_count, t_per_anchor)
    )
    per_anchor = per_anchor * valid
    k_max = int(per_anchor.max()) if len(per_anchor) else 0
    if k_max == 0:
        empty = torch.empty(0, dtype=torch.long, device=labels.device)
        return empty, empty.clone(), empty.clone()
    anchors = torch.where(valid)[0]
    positives = torch.multinomial(pos_probs[anchors], k_max, replacement=True)
    negatives = torch.multinomial(neg_probs[anchors], k_max, replacement=True)
    keep = torch.arange(k_max, device=labels.device) < per_anchor[anchors].unsqueeze(1)
    anchor_grid = anchors.unsqueeze(1).expand(-1, k_max)
    return anchor_grid[keep], positives[keep], negatives[keep]


def convert_to_triplets(
    indices_tuple: IndicesTuple | None,
    labels: Tensor | None,
    ref_labels: Tensor | None = None,
    t_per_anchor: TripletsPerAnchor = 100,
) -> TripletIndices:
    if indices_tuple is None:
        if labels is None:
            raise ValueError("labels are required when indices_tuple is None")
        if t_per_anchor == "all":
            return get_all_triplets_indices(labels, ref_labels)
        return get_random_triplet_indices(labels, ref_labels, t_per_anchor=t_per_anchor)
    if len(indices_tuple) == 3:
        return indices_tuple
    a1, p, a2, n = indices_tuple
    p_idx, n_idx = torch.where(a1.unsqueeze(1) == a2)
    return a1[p_idx], p[p_idx], n[n_idx]


def convert_to_weights(
    indices_tuple: IndicesTuple | None,
    labels: Tensor,
    dtype: torch.dtype,
    using_ref: bool = False,
) -> Tensor:
    weights = torch.zeros(labels.shape[0], device=labels.device, dtype=dtype)
    if indices_tuple is None or all(len(x) == 0 for x in indices_tuple):
        return weights + 1
    indices = get_anchors(indices_tuple) if using_ref else torch.cat(indices_tuple)
    indices, counts = torch.unique(indices, return_counts=True)
    weights[indices] = counts.to(dtype=dtype) / torch.max(counts)
    return weights


def get_anchors(indices_tuple: IndicesTuple) -> Tensor:
    if len(indices_tuple) == 3:
        return indices_tuple[0]
    return torch.cat([indices_tuple[0], indices_tuple[2]], dim=0)


def concatenate_indices_tuples(it1: IndicesTuple, it2: IndicesTuple) -> IndicesTuple:
    joined = tuple(
        torch.cat([x, y.to(device=x.device)], dim=0)
        for x, y in zip(it1, it2, strict=True)
    )
    return cast(IndicesTuple, joined)


def remove_self_comparisons(
    indices_tuple: IndicesTuple,
    curr_batch_idx: Tensor,
    ref_size: int,
    ref_is_subset: bool = False,
) -> IndicesTuple:
    if len(indices_tuple) not in (3, 4):
        raise ValueError(
            "indices_tuple must contain three triplet tensors or four pair tensors"
        )
    if len(curr_batch_idx) == 0:
        return indices_tuple
    s, e = curr_batch_idx[0], curr_batch_idx[-1]
    keep_mask = not_self_comparisons(
        indices_tuple[0],
        indices_tuple[1],
        s,
        e,
        curr_batch_idx,
        ref_size,
        ref_is_subset,
    )
    if len(indices_tuple) == 3:
        a, p, n = indices_tuple
        return a[keep_mask], p[keep_mask], n[keep_mask]
    a1, p, a2, n = indices_tuple
    return a1[keep_mask], p[keep_mask], a2, n


def not_self_comparisons(
    a: Tensor,
    p: Tensor,
    s: Tensor,
    e: Tensor,
    curr_batch_idx: Tensor,
    ref_size: int,
    ref_is_subset: bool = False,
) -> Tensor:
    if ref_is_subset:
        a, p = p, a
    curr_batch = torch.any(p.unsqueeze(1) == curr_batch_idx, dim=1)
    a_c = a[curr_batch]
    p_c = p[curr_batch] - s
    if e <= s:
        p_c[p_c <= e - s] += ref_size
    without_self_comparisons = curr_batch.clone()
    without_self_comparisons[torch.where(curr_batch)[0][a_c == p_c]] = False
    return without_self_comparisons | ~curr_batch
