from .threshold_reducer import ThresholdReducer

class AvgNonZeroReducer(ThresholdReducer):
    def __init__(self):
        super().__init__(threshold=0)