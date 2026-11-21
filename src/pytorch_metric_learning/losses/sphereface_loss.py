import torch
from torch import Tensor

from ..utils import _common
from .large_margin_softmax_loss import LargeMarginSoftmaxLoss


class SphereFaceLoss(LargeMarginSoftmaxLoss):
    def scale_logits(self, logits: Tensor, embeddings: Tensor) -> Tensor:
        dtype = torch.promote_types(logits.dtype, embeddings.dtype)
        work_dtype = _common.accumulation_dtype(dtype)
        embedding_norms = self.distance.get_norm(embeddings.to(work_dtype))
        return (logits.to(work_dtype) * embedding_norms.unsqueeze(1) * self.scale).to(
            dtype
        )
