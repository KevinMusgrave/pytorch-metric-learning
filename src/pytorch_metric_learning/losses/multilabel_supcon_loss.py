import torch

from ..distances import CosineSimilarity
from ..reducers import AvgNonZeroReducer
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from ..utils.module_with_records import ModuleWithRecords
from .generic_pair_loss import GenericPairLoss
from .base_loss_wrapper import BaseLossWrapper

# adapted from https://github.com/HobbitLong/SupContrast
# modified for multi-supcon
class MultiSupConLoss(GenericPairLoss):
    def __init__(self, num_classes, temperature=0.1, threshold=0.3, **kwargs):
        super().__init__(mat_based_loss=True, **kwargs)
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)
        self.num_classes = num_classes
        self.threshold = threshold

    def _compute_loss(self, mat, pos_mask, neg_mask, multi_val):
        if pos_mask.bool().any() and neg_mask.bool().any():
            # if dealing with actual distances, use negative distances
            if not self.distance.is_inverted:
                mat = -mat
            mat = mat / self.temperature
            mat_max, _ = mat.max(dim=1, keepdim=True)
            mat = mat - mat_max.detach()  # for numerical stability

            denominator = lmu.logsumexp(
                mat, keep_mask=(pos_mask + neg_mask).bool(), add_one=False, dim=1
            )
            log_prob = mat - denominator
            mean_log_prob_pos = (multi_val * log_prob * pos_mask).sum(dim=1) / (
                pos_mask.sum(dim=1) + c_f.small_val(mat.dtype)
            )

            return {
                "loss": {
                    "losses": -mean_log_prob_pos,
                    "indices": c_f.torch_arange_from_size(mat),
                    "reduction_type": "element",
                }
            }
        return self.zero_losses()

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def get_default_distance(self):
        return CosineSimilarity()

    #  ==== class methods below are overriden for adaptability to multi-supcon ====

    def mat_based_loss(self, mat, indices_tuple):
        a1, p, a2, n, jaccard_mat = indices_tuple
        pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        return self._compute_loss(mat, pos_mask, neg_mask, jaccard_mat)
    
    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = convert_to_pairs(
            indices_tuple, 
            labels, 
            ref_labels, 
            threshold=self.threshold)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        mat = self.distance(embeddings, ref_emb)
        return self.loss_method(mat, indices_tuple)

    def forward(
        self, embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None
    ):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
        Returns: the loss
        """
        self.reset_stats()
        check_shapes_multilabels(embeddings, labels)
        ref_emb, ref_labels = set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        loss_dict = self.compute_loss(
            embeddings, labels, indices_tuple, ref_emb, ref_labels
        )
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)
    
    # =========================================================================


# ================== cross batch memory for multi-supcon ==================
class CrossBatchMemory4MultiLabel(BaseLossWrapper, ModuleWithRecords):
    def __init__(self, loss, embedding_size, memory_size=1024, miner=None, **kwargs):
        super().__init__(loss=loss, **kwargs)
        self.loss = loss
        self.miner = miner
        self.embedding_size = embedding_size
        self.memory_size = memory_size
        self.num_classes = loss.num_classes
        self.reset_queue()
        self.add_to_recordable_attributes(
            list_of_names=["embedding_size", "memory_size", "queue_idx"], is_stat=False
        )

    @staticmethod
    def supported_losses():
        return [
            "MultiSupConLoss"
        ]

    @classmethod
    def check_loss_support(cls, loss_name):
        if loss_name not in cls.supported_losses():
            raise Exception(f"CrossBatchMemory not supported for {loss_name}")

    def forward(self, embeddings, labels, indices_tuple=None, enqueue_mask=None):
        if indices_tuple is not None and enqueue_mask is not None:
            raise ValueError("indices_tuple and enqueue_mask are mutually exclusive")
        if enqueue_mask is not None:
            assert len(enqueue_mask) == len(embeddings)
        else:
            assert len(embeddings) <= len(self.embedding_memory)
        self.reset_stats()
        device = embeddings.device
        labels = c_f.to_device(labels, device=device)
        self.embedding_memory = c_f.to_device(
            self.embedding_memory, device=device, dtype=embeddings.dtype
        )
        self.label_memory = c_f.to_device(
            self.label_memory, device=device, dtype=labels.dtype
        )

        if enqueue_mask is not None:
            emb_for_queue = embeddings[enqueue_mask]
            labels_for_queue = labels[enqueue_mask]
            embeddings = embeddings[~enqueue_mask]
            labels = labels[~enqueue_mask]
            do_remove_self_comparisons = False
        else:
            emb_for_queue = embeddings
            labels_for_queue = labels
            do_remove_self_comparisons = True

        queue_batch_size = len(emb_for_queue)
        self.add_to_memory(emb_for_queue, labels_for_queue, queue_batch_size)

        if not self.has_been_filled:
            E_mem = self.embedding_memory[: self.queue_idx]
            L_mem = self.label_memory[: self.queue_idx]
        else:
            E_mem = self.embedding_memory
            L_mem = self.label_memory

        indices_tuple = self.create_indices_tuple(
            embeddings,
            labels,
            E_mem,
            L_mem,
            indices_tuple,
            do_remove_self_comparisons,
        )
        loss = self.loss(embeddings, labels, indices_tuple, E_mem, L_mem)
        return loss

    def add_to_memory(self, embeddings, labels, batch_size):
        self.curr_batch_idx = (
            torch.arange(
                self.queue_idx, self.queue_idx + batch_size, device=labels.device
            )
            % self.memory_size
        )
        self.embedding_memory[self.curr_batch_idx] = embeddings.detach()
        self.label_memory[self.curr_batch_idx] = labels.detach()
        prev_queue_idx = self.queue_idx
        self.queue_idx = (self.queue_idx + batch_size) % self.memory_size
        if (not self.has_been_filled) and (self.queue_idx <= prev_queue_idx):
            self.has_been_filled = True

    def create_indices_tuple(
        self,
        embeddings,
        labels,
        E_mem,
        L_mem,
        input_indices_tuple,
        do_remove_self_comparisons,
    ):
        if self.miner:
            indices_tuple = self.miner(embeddings, labels, E_mem, L_mem)
        else:
            indices_tuple = get_all_pairs_indices(labels, L_mem)

        if do_remove_self_comparisons:
            indices_tuple = remove_self_comparisons(
                indices_tuple, self.curr_batch_idx, self.memory_size
            )

        if input_indices_tuple is not None:
            if len(input_indices_tuple) == 3 and len(indices_tuple) == 4:
                input_indices_tuple = convert_to_pairs(input_indices_tuple, labels)
            elif len(input_indices_tuple) == 4 and len(indices_tuple) == 3:
                input_indices_tuple = convert_to_triplets(
                    input_indices_tuple, labels
                )
            indices_tuple = c_f.concatenate_indices_tuples(
                indices_tuple, input_indices_tuple
            )

        return indices_tuple

    def reset_queue(self):
        self.register_buffer(
            "embedding_memory", torch.zeros(self.memory_size, self.embedding_size)
        )
        self.register_buffer(
            "label_memory", torch.zeros(self.memory_size, self.num_classes)
        )
        self.has_been_filled = False
        self.queue_idx = 0

# =========================================================================

# compute jaccard similarity
def jaccard(labels, ref_labels=None):
    if ref_labels is None:
        ref_labels = labels
        
    labels1 = labels.float()
    labels2 = ref_labels.float()

    # compute jaccard similarity
    # jaccard = intersection / union 
    labels1_union = labels1.sum(-1)
    labels2_union = labels2.sum(-1)
    union = labels1_union.unsqueeze(1) + labels2_union.unsqueeze(0)
    intersection = torch.mm(labels1, labels2.T)
    jaccard_matrix = intersection / (union - intersection)
    
    # return indices of jaccard similarity above threshold
    return jaccard_matrix

# ====== methods below are overriden for adaptability to multi-supcon ======

# use jaccard similarity to get matches
def get_matches_and_diffs(labels, ref_labels=None, threshold=0.3):
    if ref_labels is None:
        ref_labels = labels
    jaccard_matrix = jaccard(labels, ref_labels)
    matches = torch.where(jaccard_matrix > threshold, 1, 0)
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    return matches, diffs, jaccard_matrix

def check_shapes_multilabels(embeddings, labels):
    if labels is not None and embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Number of embeddings must equal number of labels")
    if labels is not None and labels.ndim != 2:
        raise ValueError("labels must be a 1D tensor of shape (batch_size,)")


def set_ref_emb(embeddings, labels, ref_emb, ref_labels):
    if ref_emb is None:
        ref_emb, ref_labels = embeddings, labels
    check_shapes_multilabels(ref_emb, ref_labels)
    return ref_emb, ref_labels


def convert_to_pairs(indices_tuple, labels, ref_labels=None, threshold=0.3):
    """
    This returns anchor-positive and anchor-negative indices,
    regardless of what the input indices_tuple is
    Args:
        indices_tuple: tuple of tensors. Each tensor is 1d and specifies indices
                        within a batch
        labels: a tensor which has the label for each element in a batch
    """
    if indices_tuple is None:
        return get_all_pairs_indices(labels, ref_labels, threshold=threshold)
    elif len(indices_tuple) == 5:
        return indices_tuple
    else:
        a, p, n, jaccard_mat = indices_tuple
        return a, p, a, n,jaccard_mat


def get_all_pairs_indices(labels, ref_labels=None, threshold=0.3):
    """
    Given a tensor of labels, this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    matches, diffs, multi_val = get_matches_and_diffs(labels, ref_labels, threshold=threshold)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx, multi_val


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


def remove_self_comparisons(
    indices_tuple, curr_batch_idx, ref_size, ref_is_subset=False
):
    # remove self-comparisons
    assert len(indices_tuple) in [4, 5]
    s, e = curr_batch_idx[0], curr_batch_idx[-1]
    if len(indices_tuple) == 4:
        a, p, n, jaccard_mat = indices_tuple
        keep_mask = lmu.not_self_comparisons(
            a, p, s, e, curr_batch_idx, ref_size, ref_is_subset
        )
        a = a[keep_mask]
        p = p[keep_mask]
        n = n[keep_mask]
        assert len(a) == len(p) == len(n)
        return a, p, n, jaccard_mat
    elif len(indices_tuple) == 5:
        a1, p, a2, n, jaccard_mat = indices_tuple
        keep_mask = lmu.not_self_comparisons(
            a1, p, s, e, curr_batch_idx, ref_size, ref_is_subset
        )
        a1 = a1[keep_mask]
        p = p[keep_mask]
        assert len(a1) == len(p)
        assert len(a2) == len(n)
        return a1, p, a2, n, jaccard_mat

# =========================================================================