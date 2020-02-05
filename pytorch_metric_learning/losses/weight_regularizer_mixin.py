class WeightRegularizerMixin:
    def __init__(self, regularizer=None, reg_weight=1, **kwargs):
        super().__init__(**kwargs)
        self.regularizer = regularizer
        self.reg_weight = reg_weight
        self.initialize_regularizer()

    def regularization_loss(self, weights):
        return self.regularizer(weights) * self.reg_weight

    def initialize_regularizer(self):
        if self.regularizer is None:
            def regularizer(weights):
                return 0
            self.regularizer = regularizer