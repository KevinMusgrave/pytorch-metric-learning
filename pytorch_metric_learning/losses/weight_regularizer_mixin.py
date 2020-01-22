class WeightRegularizerMixin:
    def __init__(self, regularizer=None, reg_weight=1, **kwargs):
        super().__init__(**kwargs)
        self.regularizer = (lambda x: 0) if regularizer is None else regularizer
        self.reg_weight = reg_weight

    def regularization_loss(self, weights):
        return self.regularizer(weights) * self.reg_weight