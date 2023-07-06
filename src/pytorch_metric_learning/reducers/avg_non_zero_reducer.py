from .threshold_reducer import ThresholdReducer


class AvgNonZeroReducer(ThresholdReducer):
    """Equivalent to ThresholdReducer with `low=0`"""
    def __init__(self, **kwargs):
        super().__init__(low=0, **kwargs)
