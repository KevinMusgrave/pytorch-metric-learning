import torch
import numpy as np
import math
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
    a = torch.arange(x.size(dim)).to(x.device)
    b = torch.arange(y.size(dim)).to(y.device)
    return torch.meshgrid(a, b)


def get_all_pairs_indices(labels, ref_labels=None):
    """
    Given a tensor of labels, this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx


def convert_to_pairs(indices_tuple, labels):
    """
    This returns anchor-positive and anchor-negative indices,
    regardless of what the input indices_tuple is
    Args:
        indices_tuple: tuple of tensors. Each tensor is 1d and specifies indices
                        within a batch
        labels: a tensor which has the label for each element in a batch
    """
    if indices_tuple is None:
        return get_all_pairs_indices(labels)
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
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
    return torch.where(triplets)


# sample triplets, with a weighted distribution if weights is specified.
def get_random_triplet_indices(
    labels, ref_labels=None, t_per_anchor=None, weights=None
):
    a_idx, p_idx, n_idx = [], [], []
    labels_device = labels.device
    ref_labels = labels if ref_labels is None else ref_labels
    ref_labels_is_labels = ref_labels is labels
    labels = labels.cpu().numpy()
    ref_labels = ref_labels.cpu().numpy()
    batch_size = ref_labels.shape[0]
    indices = np.arange(batch_size)
    for i, label in enumerate(labels):
        all_pos_pair_mask = ref_labels == label
        if ref_labels_is_labels:
            all_pos_pair_mask &= indices != i
        all_pos_pair_idx = np.where(all_pos_pair_mask)[0]
        curr_label_count = len(all_pos_pair_idx)
        if curr_label_count == 0:
            continue
        k = curr_label_count if t_per_anchor is None else t_per_anchor

        if weights is not None and not np.any(np.isnan(weights[i])):
            n_idx += c_f.NUMPY_RANDOM.choice(batch_size, k, p=weights[i]).tolist()
        else:
            possible_n_idx = list(np.where(ref_labels != label)[0])
            n_idx += c_f.NUMPY_RANDOM.choice(possible_n_idx, k).tolist()

        a_idx.extend([i] * k)
        curr_p_idx = c_f.safe_random_choice(all_pos_pair_idx, k)
        p_idx.extend(curr_p_idx.tolist())

    a_idx = torch.LongTensor(a_idx).to(labels_device)
    p_idx = torch.LongTensor(p_idx).to(labels_device)
    n_idx = torch.LongTensor(n_idx).to(labels_device)
    return a_idx, p_idx, n_idx


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


def convert_to_triplets(indices_tuple, labels, t_per_anchor=100):
    """
    This returns anchor-positive-negative triplets
    regardless of what the input indices_tuple is
    """
    if indices_tuple is None:
        if t_per_anchor == "all":
            return get_all_triplets_indices(labels)
        else:
            return get_random_triplet_indices(labels, t_per_anchor=t_per_anchor)
    elif len(indices_tuple) == 3:
        return indices_tuple
    else:
        a_out, p_out, n_out = [], [], []
        a1, p, a2, n = indices_tuple
        empty_output = [torch.tensor([]).to(labels.device)] * 3
        if len(a1) == 0 or len(a2) == 0:
            return empty_output
        for i in range(len(labels)):
            pos_idx = torch.where(a1 == i)[0]
            neg_idx = torch.where(a2 == i)[0]
            if len(pos_idx) > 0 and len(neg_idx) > 0:
                p_idx = p[pos_idx]
                n_idx = n[neg_idx]
                p_idx, n_idx = matched_size_indices(p_idx, n_idx)
                a_idx = torch.ones_like(c_f.longest_list([p_idx, n_idx])) * i
                a_out.append(a_idx)
                p_out.append(p_idx)
                n_out.append(n_idx)
        try:
            return [torch.cat(x, dim=0) for x in [a_out, p_out, n_out]]
        except RuntimeError:
            # assert that the exception was caused by disjoint a1 and a2
            # otherwise something has gone wrong
            assert len(np.intersect1d(a1, a2)) == 0
            return empty_output


def convert_to_weights(indices_tuple, labels, dtype):
    """
    Returns a weight for each batch element, based on
    how many times they appear in indices_tuple.
    """
    weights = torch.zeros_like(labels).type(dtype)
    if indices_tuple is None:
        return weights + 1
    if all(len(x) == 0 for x in indices_tuple):
        return weights + 1
    indices, counts = torch.unique(torch.cat(indices_tuple, dim=0), return_counts=True)
    counts = counts.type(dtype) / torch.sum(counts)
    weights[indices] = counts / torch.max(counts)
    return weights
