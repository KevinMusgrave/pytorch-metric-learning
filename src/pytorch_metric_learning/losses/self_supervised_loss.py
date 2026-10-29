import torch
from torch import Tensor

from .base_metric_loss_function import BaseMetricLossFunction


class SelfSupervisedLoss(torch.nn.Module):
    def __init__(self, loss: BaseMetricLossFunction, symmetric: bool = True) -> None:
        super().__init__()
        if not isinstance(loss, BaseMetricLossFunction):
            raise TypeError("SelfSupervisedLoss requires a BaseMetricLossFunction")
        self.loss = loss
        self.symmetric = symmetric

    def forward(self, embeddings: Tensor, ref_emb: Tensor) -> Tensor:
        labels = torch.arange(embeddings.shape[0], device=embeddings.device)
        if self.symmetric:
            embeddings = torch.cat([embeddings, ref_emb], dim=0)
            labels = torch.cat([labels, labels], dim=0)
            return self.loss(embeddings, labels)
        return self.loss(embeddings, labels, ref_emb=ref_emb, ref_labels=labels.clone())
