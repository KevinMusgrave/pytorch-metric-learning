import torch

from ..distances import LpDistance
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


class NCALoss(BaseMetricLossFunction):
    def __init__(self, softmax_scale=1, **kwargs):
        super().__init__(**kwargs)
        self.softmax_scale = softmax_scale
        self.add_to_recordable_attributes(
            list_of_names=["softmax_scale"], is_stat=False
        )

    # https://www.cs.toronto.edu/~hinton/absps/nca.pdf
    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        if len(embeddings) <= 1:
            return self.zero_losses()
        c_f.labels_required(labels)
        return self.nca_computation(
            embeddings, ref_emb, labels, ref_labels, indices_tuple
        )

    def nca_computation(
        self, query, reference, query_labels, reference_labels, indices_tuple
    ):
        dtype = query.dtype
        miner_weights = lmu.convert_to_weights(
            indices_tuple, query_labels, dtype=dtype, using_ref=query is not reference
        )
        mat = self.distance(query, reference)
        if not self.distance.is_inverted:
            mat = -mat
        if query is reference:
            mat.fill_diagonal_(c_f.neg_inf(dtype))
        same_labels = c_f.to_dtype(
            query_labels.unsqueeze(1) == reference_labels.unsqueeze(0), dtype=dtype
        )
        exp = torch.nn.functional.softmax(self.softmax_scale * mat, dim=1)
        exp = torch.sum(exp * same_labels, dim=1)
        non_zero = exp != 0
        loss = -torch.log(exp[non_zero]) * miner_weights[non_zero]
        return {
            "loss": {
                "losses": loss,
                "indices": c_f.torch_arange_from_size(query)[non_zero],
                "reduction_type": "element",
            }
        }

    def get_default_distance(self):
        return LpDistance(power=2)
