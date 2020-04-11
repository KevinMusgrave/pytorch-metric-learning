from .weight_regularizer_mixin import WeightRegularizerMixin
from .base_metric_loss_function import BaseMetricLossFunction
import torch
from ..utils import loss_and_miner_utils as lmu

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
        
    def compute_loss(self, embeddings, labels, indices_tuple):
        miner_weights = lmu.convert_to_weights(indices_tuple, labels).unsqueeze(1)
        prox = torch.nn.functional.normalize(self.proxies, p=2, dim=1) if self.normalize_embeddings else self.proxies
        cos = lmu.sim_mat(embeddings, prox)

        if not self.normalize_embeddings:
            embedding_norms_mat = self.embedding_norms.unsqueeze(0)*torch.norm(prox, p=2, dim=1, keepdim=True)
            cos = cos / (embedding_norms_mat.t())

        pos_mask = torch.nn.functional.one_hot(labels, self.num_classes).float()
        neg_mask = 1 - pos_mask

        with_pos_proxies = torch.nonzero(torch.sum(pos_mask, dim=0) != 0).squeeze(1)

        pos_term = lmu.logsumexp(-self.alpha * (cos - self.margin), keep_mask=pos_mask*miner_weights, add_one=True, dim=0)
        neg_term = lmu.logsumexp(self.alpha * (cos + self.margin), keep_mask=neg_mask*miner_weights, add_one=True, dim=0)
        
        pos_term = torch.sum(pos_term) / len(with_pos_proxies)
        neg_term = torch.sum(neg_term) / self.num_classes
           
        return pos_term + neg_term + self.regularization_loss(self.proxies)
