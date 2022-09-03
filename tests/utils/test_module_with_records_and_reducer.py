import unittest

import torch

from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.reducers import AvgNonZeroReducer

from .. import WITH_COLLECT_STATS


class TestModuleWithRecordsAndReducer(unittest.TestCase):
    def test_deepcopy_reducer(self):
        loss_fn = ContrastiveLoss(
            pos_margin=0, neg_margin=2, reducer=AvgNonZeroReducer()
        )
        embeddings = torch.randn(128, 64)
        labels = torch.randint(low=0, high=10, size=(128,))
        loss_fn(embeddings, labels)

        if WITH_COLLECT_STATS:
            self.assertTrue(
                loss_fn.reducer.reducers["pos_loss"].pos_pairs_past_filter > 0
            )
            self.assertTrue(
                loss_fn.reducer.reducers["neg_loss"].neg_pairs_past_filter > 0
            )
