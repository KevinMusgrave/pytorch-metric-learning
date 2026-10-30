from .threshold_reducer import ThresholdReducer


class AvgNonZeroReducer(ThresholdReducer):
    def __init__(self) -> None:
        super().__init__(low=0)
