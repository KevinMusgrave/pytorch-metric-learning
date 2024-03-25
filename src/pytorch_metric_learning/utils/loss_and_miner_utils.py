import math

import numpy as np
import torch

from . import common_functions as c_f


# input must be 2D
def logsumexp(x, keep_mask=None, add_one=True, dim=1):
    if keep_mask is not None:
        x = x.masked_fill(~keep_mask, c_f.neg_inf(x.dtype))
    if add_one:
        zeros = torch.zeros(x.size(dim - 1), dtype=x.dtype, device=x.device).unsqueeze(
            dim
        )
        x = torch.cat([x, zeros], dim=dim)

    output = torch.logsumexp(x, dim=dim, keepdim=True)
    if keep_mask is not None:
        output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
    return output


def meshgrid_from_sizes(x, y, dim=0):
    a = torch.arange(x.size(dim), device=x.device)
    b = torch.arange(y.size(dim), device=y.device)
    return torch.meshgrid(a, b, indexing="ij")


def get_matches_and_diffs(labels, ref_labels=None):
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    return matches, diffs


def get_all_pairs_indices(labels, ref_labels=None):
    """
    Given a tensor of labels, this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    matches, diffs = get_matches_and_diffs(labels, ref_labels)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx


def convert_to_pairs(indices_tuple, labels, ref_labels=None):
    """
    This returns anchor-positive and anchor-negative indices,
    regardless of what the input indices_tuple is
    Args:
        indices_tuple: tuple of tensors. Each tensor is 1d and specifies indices
                        within a batch
        labels: a tensor which has the label for each element in a batch
    """
    if indices_tuple is None:
        return get_all_pairs_indices(labels, ref_labels)
    elif len(indices_tuple) == 4:
        return indices_tuple
    else:
        a, p, n = indices_tuple
        return a, p, a, n


def convert_to_pos_pairs_with_unique_labels(indices_tuple, labels):
    a, p, _, _ = convert_to_pairs(indices_tuple, labels)
    _, unique_idx = np.unique(labels[a].cpu().numpy(), return_index=True)
    return a[unique_idx], p[unique_idx]


def pos_pairs_from_tuple(indices_tuple):
    return indices_tuple[:2]


def neg_pairs_from_tuple(indices_tuple):
    return indices_tuple[2:]


def get_all_triplets_indices(labels, ref_labels=None):
    all_matches, all_diffs = get_matches_and_diffs(labels, ref_labels)

    if (all_matches.shape[0] * all_matches.shape[1] * all_matches.shape[1]
            < torch.iinfo(torch.int32).max):
        # torch.nonzero is not supported for tensors with more than INT_MAX elements
        triplets = all_matches.unsqueeze(2) * all_diffs.unsqueeze(1)
        return torch.where(triplets)

    all_matches, all_diffs = all_matches.bool(), all_diffs.bool()

    # Find anchors with at least a positive and a negative
    indices = torch.arange(0, len(labels), device=labels.device)
    indices = indices[all_matches.any(dim=1) & all_diffs.any(dim=1)]

    # No triplets found
    if len(indices) == 0:
        return (torch.tensor([], device=labels.device, dtype=labels.dtype),
                torch.tensor([], device=labels.device, dtype=labels.dtype),
                torch.tensor([], device=labels.device, dtype=labels.dtype))

    # Compute all triplets
    anchors = []
    positives = []
    negatives = []
    for i in indices:
        matches = all_matches[i].nonzero(as_tuple=False).squeeze(1)
        diffs = all_diffs[i].nonzero(as_tuple=False).squeeze(1)
        nd = len(diffs)
        nm = len(matches)
        matches = matches.repeat_interleave(nd)
        diffs = diffs.repeat(nm)
        anchors.append(torch.full((len(matches),), i, dtype=labels.dtype, device=labels.device))
        positives.append(matches)
        negatives.append(diffs)
    return torch.cat(anchors), torch.cat(positives), torch.cat(negatives)


# sample triplets, with a weighted distribution if weights is specified.
def get_random_triplet_indices(
    labels, ref_labels=None, t_per_anchor=None, weights=None
):
    a_idx, p_idx, n_idx = [], [], []
    labels_device = labels.device
    ref_labels = labels if ref_labels is None else ref_labels
    unique_labels = torch.unique(labels)
    for label in unique_labels:
        # Get indices of positive samples for this label.
        p_inds = torch.where(ref_labels == label)[0]
        if ref_labels is labels:
            a_inds = p_inds
        else:
            a_inds = torch.where(labels == label)[0]
        n_inds = torch.where(ref_labels != label)[0]
        n_a = len(a_inds)
        n_p = len(p_inds)
        min_required_p = 2 if ref_labels is labels else 1
        if (n_p < min_required_p) or (len(n_inds) < 1):
            continue

        k = n_p if t_per_anchor is None else t_per_anchor
        num_triplets = n_a * k
        p_inds_ = p_inds.expand((n_a, n_p))
        # Remove anchors from list of possible positive samples.
        if ref_labels is labels:
            p_inds_ = p_inds_[~torch.eye(n_a).bool()].view((n_a, n_a - 1))
        # Get indices of indices of k random positive samples for each anchor.
        p_ = torch.randint(0, p_inds_.shape[1], (num_triplets,))
        # Get indices of indices of corresponding anchors.
        a_ = torch.arange(n_a).view(-1, 1).repeat(1, k).view(num_triplets)
        p = p_inds_[a_, p_]
        a = a_inds[a_]

        # Get indices of negative samples for this label.
        if weights is not None:
            w = weights[:, n_inds][a]
            non_zero_rows = torch.where(torch.sum(w, dim=1) > 0)[0]
            if len(non_zero_rows) == 0:
                continue
            w = w[non_zero_rows]
            a = a[non_zero_rows]
            p = p[non_zero_rows]
            # Sample the negative indices according to the weights.
            if w.dtype == torch.float16:
                # special case needed due to pytorch cuda bug
                # https://github.com/pytorch/pytorch/issues/19900
                w = w.type(torch.float32)
            n_ = torch.multinomial(w, 1, replacement=True).flatten()
        else:
            # Sample the negative indices uniformly.
            n_ = torch.randint(0, len(n_inds), (num_triplets,))
        n = n_inds[n_]
        a_idx.append(a)
        p_idx.append(p)
        n_idx.append(n)

    if len(a_idx) > 0:
        a_idx = c_f.to_device(torch.cat(a_idx), device=labels_device, dtype=torch.long)
        p_idx = c_f.to_device(torch.cat(p_idx), device=labels_device, dtype=torch.long)
        n_idx = c_f.to_device(torch.cat(n_idx), device=labels_device, dtype=torch.long)
        assert len(a_idx) == len(p_idx) == len(n_idx)
        return a_idx, p_idx, n_idx
    else:
        empty = torch.tensor([], device=labels_device, dtype=torch.long)
        return empty.clone(), empty.clone(), empty.clone()


def repeat_to_match_size(smaller_set, larger_size, smaller_size):
    num_repeat = math.ceil(float(larger_size) / float(smaller_size))
    return smaller_set.repeat(num_repeat)[:larger_size]


def matched_size_indices(curr_p_idx, curr_n_idx):
    num_pos_pairs = len(curr_p_idx)
    num_neg_pairs = len(curr_n_idx)
    if num_pos_pairs > num_neg_pairs:
        n_idx = repeat_to_match_size(curr_n_idx, num_pos_pairs, num_neg_pairs)
        p_idx = curr_p_idx
    else:
        p_idx = repeat_to_match_size(curr_p_idx, num_neg_pairs, num_pos_pairs)
        n_idx = curr_n_idx
    return p_idx, n_idx


def convert_to_triplets(indices_tuple, labels, ref_labels=None, t_per_anchor=100):
    """
    This returns anchor-positive-negative triplets
    regardless of what the input indices_tuple is
    """
    if indices_tuple is None:
        if t_per_anchor == "all":
            return get_all_triplets_indices(labels, ref_labels)
        else:
            return get_random_triplet_indices(
                labels, ref_labels, t_per_anchor=t_per_anchor
            )
    elif len(indices_tuple) == 3:
        return indices_tuple
    else:
        a1, p, a2, n = indices_tuple
        p_idx, n_idx = torch.where(a1.unsqueeze(1) == a2)
        return a1[p_idx], p[p_idx], n[n_idx]


def convert_to_weights(indices_tuple, labels, dtype, using_ref=False):
    """
    Returns a weight for each batch element, based on
    how many times they appear in indices_tuple.
    """
    weights = torch.zeros(labels.shape[0], device=labels.device)
    weights = c_f.to_dtype(weights, dtype=dtype)
    if (indices_tuple is None) or (all(len(x) == 0 for x in indices_tuple)):
        return weights + 1
    if using_ref:
        # if using ref, then only the anchors refer to the query set
        indices = get_anchors(indices_tuple)
    else:
        indices = torch.cat(indices_tuple, dim=0)
    indices, counts = torch.unique(indices, return_counts=True)
    weights[indices] = c_f.to_dtype(counts, dtype=dtype) / torch.max(counts)
    return weights


def remove_self_comparisons(
    indices_tuple, curr_batch_idx, ref_size, ref_is_subset=False
):
    # remove self-comparisons
    assert len(indices_tuple) in [3, 4]
    s, e = curr_batch_idx[0], curr_batch_idx[-1]
    if len(indices_tuple) == 3:
        a, p, n = indices_tuple
        keep_mask = not_self_comparisons(
            a, p, s, e, curr_batch_idx, ref_size, ref_is_subset
        )
        a = a[keep_mask]
        p = p[keep_mask]
        n = n[keep_mask]
        assert len(a) == len(p) == len(n)
        return a, p, n
    elif len(indices_tuple) == 4:
        a1, p, a2, n = indices_tuple
        keep_mask = not_self_comparisons(
            a1, p, s, e, curr_batch_idx, ref_size, ref_is_subset
        )
        a1 = a1[keep_mask]
        p = p[keep_mask]
        assert len(a1) == len(p)
        assert len(a2) == len(n)
        return a1, p, a2, n


# a: anchors
# p: positives
# s: curr batch start idx in queue
# e: curr batch end idx in queue
def not_self_comparisons(a, p, s, e, curr_batch_idx, ref_size, ref_is_subset=False):
    if ref_is_subset:
        a, p = p, a
    curr_batch = torch.any(p.unsqueeze(1) == curr_batch_idx, dim=1)
    a_c = a[curr_batch]
    p_c = p[curr_batch]
    p_c -= s
    if e <= s:
        p_c[p_c <= e - s] += ref_size
    without_self_comparisons = curr_batch.clone()
    without_self_comparisons[torch.where(curr_batch)[0][a_c == p_c]] = False
    return without_self_comparisons | ~curr_batch


def get_anchors(indices_tuple):
    if len(indices_tuple) == 3:
        return indices_tuple[0]
    elif len(indices_tuple) == 4:
        return torch.cat([indices_tuple[0], indices_tuple[2]], dim=0)
