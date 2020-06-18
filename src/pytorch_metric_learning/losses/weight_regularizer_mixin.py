class WeightRegularizerMixin:
    def __init__(self, regularizer=None, reg_weight=1, **kwargs):
        super().__init__(**kwargs)
        self.regularizer = regularizer
        self.reg_weight = reg_weight

    def regularization_loss(self, weights):
        if self.regularizer is None:
            loss = 0
        else:
            loss = self.regularizer(weights) * self.reg_weight
        return {"losses": loss, "indices": None, "reduction_type": "already_reduced"}