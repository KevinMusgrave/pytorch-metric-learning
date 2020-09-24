from .mixins import WeightRegularizerMixin
from .base_metric_loss_function import BaseMetricLossFunction
import torch
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f
from ..reducers import DivisorReducer
from ..distances import CosineSimilarity

# adapted from
# https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/blob/master/code/losses.py
# https://github.com/geonm/proxy-anchor-loss/blob/master/pytorch-proxy-anchor.py
# suggested in this issue: https://github.com/KevinMusgrave/pytorch-metric-learning/issues/32
class ProxyAnchorLoss(WeightRegularizerMixin, BaseMetricLossFunction):
    def __init__(self, num_classes, embedding_size, margin=0.1, alpha=32, **kwargs):
        super().__init__(**kwargs)
        self.proxies = torch.nn.Parameter(torch.Tensor(num_classes, embedding_size))
        self.weight_init_func(self.proxies)
        self.num_classes = num_classes
        self.margin = margin
        self.alpha = alpha
        self.add_to_recordable_attributes(
            list_of_names=["num_classes", "alpha", "margin"], is_stat=False
        )

    def cast_types(self, dtype, device):
        self.proxies.data = self.proxies.data.to(device).type(dtype)

    def compute_loss(self, embeddings, labels, indices_tuple):
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        miner_weights = lmu.convert_to_weights(
            indices_tuple, labels, dtype=dtype
        ).unsqueeze(1)
        miner_weights = miner_weights - 1

        cos = self.distance(embeddings, self.proxies)

        pos_mask = torch.nn.functional.one_hot(labels, self.num_classes)
        neg_mask = 1 - pos_mask

        with_pos_proxies = torch.where(torch.sum(pos_mask, dim=0) != 0)[0]

        pos_exp = self.distance.margin(cos, self.margin)
        neg_exp = self.distance.margin(-self.margin, cos)

        pos_term = lmu.logsumexp(
            (self.alpha * pos_exp) + miner_weights,
            keep_mask=pos_mask.bool(),
            add_one=True,
            dim=0,
        )
        neg_term = lmu.logsumexp(
            (self.alpha * neg_exp) + miner_weights,
            keep_mask=neg_mask.bool(),
            add_one=True,
            dim=0,
        )

        loss_indices = c_f.torch_arange_from_size(self.proxies)

        loss_dict = {
            "pos_loss": {
                "losses": pos_term.squeeze(0),
                "indices": loss_indices,
                "reduction_type": "element",
                "divisor_summands": {"num_pos_proxies": len(with_pos_proxies)},
            },
            "neg_loss": {
                "losses": neg_term.squeeze(0),
                "indices": loss_indices,
                "reduction_type": "element",
                "divisor_summands": {"num_classes": self.num_classes},
            },
        }

        self.add_weight_regularization_to_loss_dict(loss_dict, self.proxies)

        return loss_dict

    def get_default_reducer(self):
        return DivisorReducer()

    def get_default_distance(self):
        return CosineSimilarity()

    def get_default_weight_init_func(self):
        return c_f.TorchInitWrapper(torch.nn.init.kaiming_normal_, mode="fan_out")

    def _sub_loss_names(self):
        return ["pos_loss", "neg_loss"]
