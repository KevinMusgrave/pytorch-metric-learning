from .threshold_reducer import ThresholdReducer

class AvgNonZeroReducer(ThresholdReducer):
    def __init__(self, **kwargs):
        super().__init__(low=0, **kwargs)