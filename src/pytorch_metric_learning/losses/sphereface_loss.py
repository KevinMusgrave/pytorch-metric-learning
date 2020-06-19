from .large_margin_softmax_loss import LargeMarginSoftmaxLoss

class SphereFaceLoss(LargeMarginSoftmaxLoss):
	# implementation of https://arxiv.org/pdf/1704.08063.pdf
    def __init__(self, normalize_weights=True, **kwargs):
        super().__init__(normalize_weights=normalize_weights, **kwargs)
        assert self.normalize_weights, "SphereFace requires that the weights are normalized"