import torch
from torch import Tensor

from ..types import IndicesTuple, LossDict
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .generic_pair_loss import GenericPairLoss


class LiftedStructureLoss(GenericPairLoss):
    def __init__(self, neg_margin=1, pos_margin=0, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        self.neg_margin = neg_margin
        self.pos_margin = pos_margin

    def compute_loss(
        self,
        embeddings: Tensor,
        labels: Tensor | None,
        indices_tuple: IndicesTuple | None,
        ref_emb: Tensor,
        ref_labels: Tensor | None,
    ) -> LossDict:
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        return super().compute_loss(
            embeddings, labels, indices_tuple, ref_emb, ref_labels
        )

    def _compute_pair_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, _ = indices_tuple

        if len(a1) > 0 and len(a2) > 0:
            pos_pairs = pos_pairs.unsqueeze(1)
            dtype = pos_pairs.dtype
            n_per_p = (
                (a2.unsqueeze(0) == a1.unsqueeze(1))
                | (a2.unsqueeze(0) == p.unsqueeze(1))
            ).to(dtype=dtype)
            neg_pairs = neg_pairs * n_per_p
            keep_mask = ~(n_per_p == 0)

            remaining_pos_margin = self.distance.margin(pos_pairs, self.pos_margin)
            remaining_neg_margin = self.distance.margin(self.neg_margin, neg_pairs)

            neg_pairs_loss = lmu.logsumexp(
                remaining_neg_margin, keep_mask=keep_mask, add_one=False, dim=1
            )
            loss_per_pos_pair = neg_pairs_loss + remaining_pos_margin
            loss_per_pos_pair = torch.relu(loss_per_pos_pair) ** 2
            loss_per_pos_pair /= 2
            return {
                "loss": {
                    "losses": loss_per_pos_pair,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()


class GeneralizedLiftedStructureLoss(GenericPairLoss):
    def __init__(self, neg_margin=1, pos_margin=0, **kwargs):
        super().__init__(mat_based_loss=True, **kwargs)
        self.neg_margin = neg_margin
        self.pos_margin = pos_margin

    def _compute_matrix_loss(self, mat, pos_mask, neg_mask):
        remaining_pos_margin = self.distance.margin(mat, self.pos_margin)
        remaining_neg_margin = self.distance.margin(self.neg_margin, mat)

        pos_loss = lmu.logsumexp(
            remaining_pos_margin, keep_mask=pos_mask.bool(), add_one=False
        )
        neg_loss = lmu.logsumexp(
            remaining_neg_margin, keep_mask=neg_mask.bool(), add_one=False
        )
        return {
            "loss": {
                "losses": torch.relu(pos_loss + neg_loss),
                "indices": c_f.torch_arange_from_size(mat),
                "reduction_type": "element",
            }
        }
