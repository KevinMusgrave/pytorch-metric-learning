from .threshold_reducer import ThresholdReducer


class SumReducer(ThresholdReducer):
    """It reduces the losses by summing up all the values.

    Subclass of ThresholdReducer, therefore it is possible to specify `low` and `high` hyperparameters.
    """

    def __init__(self, **kwargs):
        kwargs["collect_stats"] = True
        super().__init__(**kwargs)

    def element_reduction(self, losses, *_):
        return super().element_reduction(losses, *_) * self.num_past_filter
