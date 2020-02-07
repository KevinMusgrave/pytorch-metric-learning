#! /usr/bin/env python3

from .large_margin_softmax_loss import LargeMarginSoftmaxLoss
import numpy as np
import torch

class CosFaceLoss(LargeMarginSoftmaxLoss):
    """
    Implementation of https://arxiv.org/pdf/1801.07698.pdf
    """
    def __init__(self, scale=64, **kwargs):
        kwargs.pop("normalize_weights", None)
        super().__init__(scale=scale, normalize_weights=True, **kwargs)

    def init_margin(self):
        pass

    def modify_cosine_of_target_classes(self, cosine_of_target_classes, *args):
        self.get_angles(cosine_of_target_classes) # For analysis purposes
        return cosine_of_target_classes - self.margin
