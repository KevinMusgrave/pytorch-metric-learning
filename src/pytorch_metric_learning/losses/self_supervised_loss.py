import torch
from torch import Tensor

from ..types import MetricLoss
from .multiple_losses import _require_metric_loss


class SelfSupervisedLoss(torch.nn.Module):
    def __init__(self, loss: MetricLoss, symmetric: bool = True) -> None:
        super().__init__()
        _require_metric_loss(loss, "SelfSupervisedLoss")
        self.loss = loss
        self.symmetric = symmetric

    def forward(
        self, embeddings: Tensor, ref_emb: Tensor, *other_views: Tensor
    ) -> Tensor:
        references = (ref_emb, *other_views)
        if any(view.shape != embeddings.shape for view in references):
            raise ValueError("all views must have the same shape")
        labels = torch.arange(embeddings.shape[0], device=embeddings.device)
        ref_labels = labels.repeat(len(references))
        ref_emb = torch.cat(references, dim=0)
        if self.symmetric:
            embeddings = torch.cat([embeddings, ref_emb], dim=0)
            labels = torch.cat([labels, ref_labels], dim=0)
            return self.loss(embeddings, labels)
        return self.loss(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
