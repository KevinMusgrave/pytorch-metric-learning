from ..utils import common_functions as c_f
from .threshold_reducer import ThresholdReducer


class ClassWeightedReducer(ThresholdReducer):
    """It weights the losses with user-specified weights and then takes the average.

    Subclass of ThresholdReducer, therefore it is possible to specify `low` and `high` hyperparameters.
    """

    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.weights = weights

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        self.weights = c_f.to_device(self.weights, losses, dtype=losses.dtype)
        losses = losses * self.weights[labels[loss_indices]]
        return super().element_reduction(losses, loss_indices, embeddings, labels)
