import torch

from ..reducers import AvgNonZeroReducer
from ..utils import loss_and_miner_utils as lmu
from .generic_pair_loss import GenericPairLoss


class ContrastiveLoss(GenericPairLoss):
    def __init__(self, pos_margin=0, neg_margin=1, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

    def _compute_pair_loss(self, pos_pairs, neg_pairs, indices_tuple):
        pos_loss, neg_loss = 0, 0
        if len(pos_pairs) > 0:
            pos_loss = self.get_per_pair_loss(pos_pairs, "pos")
        if len(neg_pairs) > 0:
            neg_loss = self.get_per_pair_loss(neg_pairs, "neg")
        return {
            "pos_loss": {
                "losses": pos_loss,
                "indices": lmu.pos_pairs_from_tuple(indices_tuple),
                "reduction_type": "pos_pair",
            },
            "neg_loss": {
                "losses": neg_loss,
                "indices": lmu.neg_pairs_from_tuple(indices_tuple),
                "reduction_type": "neg_pair",
            },
        }

    def get_per_pair_loss(self, pair_dists, pos_or_neg):
        loss_calc_func = self.pos_calc if pos_or_neg == "pos" else self.neg_calc
        margin = self.pos_margin if pos_or_neg == "pos" else self.neg_margin
        per_pair_loss = loss_calc_func(pair_dists, margin)
        return per_pair_loss

    def pos_calc(self, pos_pairs, margin):
        return torch.nn.functional.relu(self.distance.margin(pos_pairs, margin))

    def neg_calc(self, neg_pairs, margin):
        return torch.nn.functional.relu(self.distance.margin(margin, neg_pairs))

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def sub_loss_names(self):
        return ["pos_loss", "neg_loss"]
