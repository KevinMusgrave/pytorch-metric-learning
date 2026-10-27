import math

import torch

from .large_margin_softmax_loss import LargeMarginSoftmaxLoss


class ArcFaceLoss(LargeMarginSoftmaxLoss):
    def __init__(
        self,
        num_classes,
        embedding_size,
        margin=28.6,
        scale=64,
        *,
        weight_regularizer=None,
        weight_reg_weight=1,
        weight_init_func=None,
        **kwargs,
    ):
        super().__init__(
            num_classes,
            embedding_size,
            margin=margin,
            scale=scale,
            weight_regularizer=weight_regularizer,
            weight_reg_weight=weight_reg_weight,
            weight_init_func=weight_init_func,
            **kwargs,
        )

    def init_margin(self):
        self.margin = math.radians(self.margin)

    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        angles = self.get_angles(cosine_of_target_classes)

        cos_theta_plus_margin = torch.cos(angles + self.margin)
        cos_theta = torch.cos(angles)

        unscaled_logits = torch.where(
            angles <= math.pi - self.margin,
            cos_theta_plus_margin,
            cos_theta - self.margin * math.sin(self.margin),
        )

        return unscaled_logits

    def scale_logits(self, logits, *_):
        return logits * self.scale
