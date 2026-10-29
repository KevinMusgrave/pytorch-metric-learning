from torch import Tensor

from .large_margin_softmax_loss import LargeMarginSoftmaxLoss


class SphereFaceLoss(LargeMarginSoftmaxLoss):
    def scale_logits(self, logits: Tensor, embeddings: Tensor) -> Tensor:
        embedding_norms = self.distance.get_norm(embeddings)
        return logits * embedding_norms.unsqueeze(1) * self.scale
