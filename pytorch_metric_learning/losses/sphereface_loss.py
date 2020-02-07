from .large_margin_softmax_loss import LargeMarginSoftmaxLoss

class SphereFaceLoss(LargeMarginSoftmaxLoss):
	# implementation of https://arxiv.org/pdf/1704.08063.pdf
    def __init__(self, **kwargs):
        kwargs.pop("normalize_weights", None)
        super().__init__(normalize_weights=True, **kwargs)