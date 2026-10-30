import unittest

from pytorch_metric_learning.losses import ContrastiveLoss, TripletMarginLoss
from pytorch_metric_learning.reducers import (
    AvgNonZeroReducer,
    MeanReducer,
    ThresholdReducer,
)


class TestSettingReducers(unittest.TestCase):
    def test_setting_reducers(self):
        for loss in [TripletMarginLoss, ContrastiveLoss]:
            for reducer in [
                ThresholdReducer(low=0),
                MeanReducer(),
                AvgNonZeroReducer(),
            ]:
                L = loss(reducer=reducer)
                assert L.reducer is reducer
