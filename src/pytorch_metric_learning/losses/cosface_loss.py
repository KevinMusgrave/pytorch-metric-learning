import torch

from ..utils import common_functions as c_f
from .large_margin_softmax_loss import LargeMarginSoftmaxLoss


class CosFaceLoss(LargeMarginSoftmaxLoss):
    """
    Implementation of https://arxiv.org/pdf/1801.07698.pdf
    """

    def __init__(self, *args, margin=0.35, scale=64, **kwargs):
        super().__init__(*args, margin=margin, scale=scale, **kwargs)

    def init_margin(self):
        pass

    def cast_types(self, dtype, device):
        self.W.data = c_f.to_device(self.W.data, device=device, dtype=dtype)

    def modify_cosine_of_target_classes(self, cosine_of_target_classes, *args):
        if self.collect_stats:
            with torch.no_grad():
                self.get_angles(
                    cosine_of_target_classes
                )  # For the purpose of collecting stats
        return cosine_of_target_classes - self.margin

    def scale_logits(self, logits, *_):
        return logits * self.scale
