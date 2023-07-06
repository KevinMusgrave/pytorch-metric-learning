import torch

from ..reducers import DivisorReducer
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


class MarginLoss(BaseMetricLossFunction):
    def __init__(
        self,
        margin=0.2,
        nu=0,
        beta=1.2,
        triplets_per_anchor="all",
        learn_beta=False,
        num_classes=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.nu = nu
        self.learn_beta = learn_beta
        self.initialize_beta(beta, num_classes)
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(
            list_of_names=["margin", "nu", "beta"], is_stat=False
        )

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = lmu.convert_to_triplets(
            indices_tuple, labels, ref_labels, self.triplets_per_anchor
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()

        beta = self.beta if len(self.beta) == 1 else self.beta[labels[anchor_idx].to("cpu")]    # When labels are on gpu gives error
        beta = c_f.to_device(beta, device=embeddings.device, dtype=embeddings.dtype)

        mat = self.distance(embeddings, ref_emb)

        d_ap = mat[anchor_idx, positive_idx]
        d_an = mat[anchor_idx, negative_idx]

        pos_loss = torch.nn.functional.relu(
            self.distance.margin(d_ap, beta) + self.margin
        )
        neg_loss = torch.nn.functional.relu(
            self.distance.margin(beta, d_an) + self.margin
        )

        num_pos_pairs = torch.sum(pos_loss > 0.0)
        num_neg_pairs = torch.sum(neg_loss > 0.0)

        divisor = num_pos_pairs + num_neg_pairs

        margin_loss = pos_loss + neg_loss

        loss_dict = {
            "margin_loss": {
                "losses": margin_loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
                "divisor": divisor,
            },
            "beta_reg_loss": self.compute_reg_loss(beta, anchor_idx, divisor),
        }

        return loss_dict

    def compute_reg_loss(self, beta, anchor_idx, divisor):
        if self.learn_beta:
            loss = beta * self.nu
            if len(self.beta) == 1:
                return {
                    "losses": loss,
                    "indices": None,
                    "reduction_type": "already_reduced",
                }
            else:
                return {
                    "losses": loss,
                    "indices": anchor_idx,
                    "reduction_type": "element",
                    "divisor": divisor,
                }
        return self.zero_loss()

    def _sub_loss_names(self):
        return ["margin_loss", "beta_reg_loss"]

    def get_default_reducer(self):
        return DivisorReducer()

    def initialize_beta(self, beta, num_classes):
        self.beta = torch.tensor([float(beta)])
        if num_classes:
            self.beta = torch.ones(num_classes) * self.beta
        if self.learn_beta:
            self.beta = torch.nn.Parameter(self.beta)
