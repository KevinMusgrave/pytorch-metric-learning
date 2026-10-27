from .large_margin_softmax_loss import LargeMarginSoftmaxLoss


class CosFaceLoss(LargeMarginSoftmaxLoss):
    def __init__(
        self,
        num_classes,
        embedding_size,
        margin=0.35,
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
        pass

    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        return cosine_of_target_classes - self.margin

    def scale_logits(self, logits, embeddings):
        return logits * self.scale
