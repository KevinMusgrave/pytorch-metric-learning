import torch

from ..distances import CosineSimilarity
from ..reducers import AvgNonZeroReducer
from ..utils import common_functions as c_f
from ..utils import multilabel_loss_and_miner_utils as mlmu
from ..utils import loss_and_miner_utils as lmu
from .generic_pair_loss import GenericPairLoss


# adapted from https://github.com/HobbitLong/SupContrast
class MultiSupConLoss(GenericPairLoss):
    def __init__(self, num_classes,  temperature=0.1, **kwargs):
        super().__init__(mat_based_loss=True, **kwargs)
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)
        self.num_classes = num_classes

    def _compute_loss(self, mat, pos_mask, neg_mask):
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
            mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (
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

    def mat_based_loss(self, mat, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        return self._compute_loss(mat, pos_mask, neg_mask)
    
    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = mlmu.convert_to_pairs(indices_tuple, labels, self.num_classes, ref_labels, device=embeddings.device)
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
        mlmu.check_shapes_multilabels(embeddings, labels)
        ref_emb, ref_labels = mlmu.set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        loss_dict = self.compute_loss(
            embeddings, labels, indices_tuple, ref_emb, ref_labels
        )
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)
    
