from .nca_loss import NCALoss
from .mixins import WeightRegularizerMixin
from ..utils import loss_and_miner_utils as lmu
import torch


class ProxyNCALoss(WeightRegularizerMixin, NCALoss):
    def __init__(self, num_classes, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.proxies = torch.nn.Parameter(torch.Tensor(num_classes, embedding_size))
        self.weight_init_func(self.proxies)
        self.proxy_labels = torch.arange(num_classes)
        self.add_to_recordable_attributes(list_of_names=["num_classes"], is_stat=False)

    def cast_types(self, dtype, device):
        self.proxies.data = self.proxies.data.to(device).type(dtype)

    def compute_loss(self, embeddings, labels, indices_tuple):
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        loss_dict = self.nca_computation(
            embeddings,
            self.proxies,
            labels,
            self.proxy_labels.to(labels.device),
            indices_tuple,
        )
        self.add_weight_regularization_to_loss_dict(loss_dict, self.proxies)
        return loss_dict
