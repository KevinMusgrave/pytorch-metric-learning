from .large_margin_softmax_loss import LargeMarginSoftmaxLoss
import torch


class SphereFaceLoss(LargeMarginSoftmaxLoss):
    # implementation of https://arxiv.org/pdf/1704.08063.pdf
    def scale_logits(self, logits, embeddings):
        embedding_norms = torch.norm(embeddings, p=2, dim=1)
        return logits * embedding_norms.unsqueeze(1) * self.scale
