import torch
from . import loss_and_miner_utils as lmu

def check_shapes_multilabels(embeddings, labels):
    if labels is not None and embeddings.shape[0] != len(labels):
        raise ValueError("Number of embeddings must equal number of labels")
    if labels is not None:
        if isinstance(labels[0], list) or isinstance(labels[0], torch.Tensor):
            pass
        else:
            raise ValueError("labels must be a list of 1d tensors or a list of lists")

def set_ref_emb(embeddings, labels, ref_emb, ref_labels):
    if ref_emb is None:
        ref_emb, ref_labels = embeddings, labels
    check_shapes_multilabels(ref_emb, ref_labels)
    return ref_emb, ref_labels

def convert_to_pairs(indices_tuple, labels, num_classes, ref_labels=None, device=None):
    """
    This returns anchor-positive and anchor-negative indices,
    regardless of what the input indices_tuple is
    Args:
        indices_tuple: tuple of tensors. Each tensor is 1d and specifies indices
                        within a batch
        labels: a tensor which has the label for each element in a batch
    """
    if indices_tuple is None:
        return get_all_pairs_indices(labels, num_classes, ref_labels, device=device)
    elif len(indices_tuple) == 4:
        return indices_tuple
    else:
        a, p, n = indices_tuple
        return a, p, a, n
    
def get_matches_and_diffs(labels, num_classes, ref_labels=None, device=None):
    matches = jaccard(num_classes, labels, ref_labels, device=device)
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    return matches, diffs


def get_all_pairs_indices(labels, num_classes, ref_labels=None, device=None):
    """
    Given a tensor of labels, this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    matches, diffs = get_matches_and_diffs(labels, num_classes, ref_labels, device)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx

def jaccard(n_classes, labels, ref_labels=None, threshold=0.3, device=torch.device("cpu")):
    if ref_labels is None:
        ref_labels = labels
    # convert multilabels to scatter labels
    labels1 = [torch.nn.functional.one_hot(torch.Tensor(label).long(), n_classes).sum(0) for label in labels]
    labels2 = [torch.nn.functional.one_hot(torch.Tensor(label).long(), n_classes).sum(0) for label in ref_labels]
    # stack and convert to float for calculation convenience
    labels1 = torch.stack(labels1).float()
    labels2 = torch.stack(labels2).float()

    # compute jaccard similarity
    # jaccard = intersection / union 
    labels1_union = labels1.sum(-1)
    labels2_union = labels2.sum(-1)
    union = labels1_union.unsqueeze(1) + labels2_union.unsqueeze(0)
    intersection = torch.mm(labels1, labels2.T)
    jaccard = intersection / (union - intersection)
    
    # return indices of jaccard similarity above threshold
    label_matrix = torch.where(jaccard > threshold, 1, 0).to(device)
    return label_matrix

def convert_to_triplets(indices_tuple, labels, ref_labels=None, t_per_anchor=100):
    """
    This returns anchor-positive-negative triplets
    regardless of what the input indices_tuple is
    """
    if indices_tuple is None:
        if t_per_anchor == "all":
            return get_all_triplets_indices(labels, ref_labels)
        else:
            return lmu.get_random_triplet_indices(
                labels, ref_labels, t_per_anchor=t_per_anchor
            )
    elif len(indices_tuple) == 3:
        return indices_tuple
    else:
        a1, p, a2, n = indices_tuple
        p_idx, n_idx = torch.where(a1.unsqueeze(1) == a2)
        return a1[p_idx], p[p_idx], n[n_idx]
    

def get_all_triplets_indices(labels, ref_labels=None):
    matches, diffs = get_matches_and_diffs(labels, ref_labels)
    triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
    return torch.where(triplets)

