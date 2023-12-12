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
                if isinstance(L, TripletMarginLoss):
                    assert type(L.reducer) is type(reducer)
                else:
                    for v in L.reducer.reducers.values():
                        assert type(v) is type(reducer)
