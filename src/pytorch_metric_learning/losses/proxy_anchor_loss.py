from .regularizer_mixins import WeightRegularizerMixin
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
    def __init__(self, num_classes, embedding_size, margin = 0.1, alpha = 32, **kwargs):
        super().__init__(**kwargs)
        self.proxies = torch.nn.Parameter(torch.randn(num_classes, embedding_size))
        torch.nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.num_classes = num_classes
        self.margin = margin
        self.alpha = alpha
        
    def cast_types(self, dtype, device):
        self.proxies.data = self.proxies.data.to(device).type(dtype)

    def compute_loss(self, embeddings, labels, indices_tuple):
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype).unsqueeze(1)

        cos = self.distance(embeddings, self.proxies)

        pos_mask = torch.nn.functional.one_hot(labels, self.num_classes).type(dtype)
        neg_mask = 1 - pos_mask

        with_pos_proxies = torch.nonzero(torch.sum(pos_mask, dim=0) != 0).squeeze(1)

        pos_exp = self.distance.margin(cos, self.margin)
        neg_exp = self.distance.margin(-self.margin, cos)

        pos_term = lmu.logsumexp(self.alpha * pos_exp, keep_mask=pos_mask*miner_weights, add_one=True, dim=0)
        neg_term = lmu.logsumexp(self.alpha * neg_exp, keep_mask=neg_mask*miner_weights, add_one=True, dim=0)

        loss_indices = c_f.torch_arange_from_size(self.proxies)

        loss_dict = {"pos_loss": {"losses": pos_term.squeeze(0), "indices": loss_indices, "reduction_type": "element", "divisor_summands": {"num_pos_proxies": len(with_pos_proxies)}},
                    "neg_loss": {"losses": neg_term.squeeze(0), "indices": loss_indices, "reduction_type": "element", "divisor_summands": {"num_classes": self.num_classes}}}
        
        self.add_weight_regularization_to_loss_dict(loss_dict, self.proxies)

        return loss_dict

    def get_default_reducer(self):
        return DivisorReducer()

    def get_default_distance(self):
        return CosineSimilarity()

    def _sub_loss_names(self):
        return ["pos_loss", "neg_loss"]