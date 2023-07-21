import numpy as np

from .threshold_reducer import ThresholdReducer


class MeanReducer(ThresholdReducer):
    """Equivalent to ThresholdReducer with default parameters.

    Any element is accepted"""

    def __init__(self, **kwargs):
        kwargs["low"] = -np.inf
        kwargs["high"] = np.inf
        super().__init__(**kwargs)
