import unittest

import torch

from pytorch_metric_learning.reducers import DoNothingReducer

from .. import TEST_DEVICE, TEST_DTYPES


class TestDoNothingReducer(unittest.TestCase):
    def test_do_nothing_reducer(self):
        reducer = DoNothingReducer()
        for dtype in TEST_DTYPES:
            loss_dict = {
                "loss": {
                    "losses": torch.randn(100).type(dtype).to(TEST_DEVICE),
                    "indices": torch.arange(100),
                    "reduction_type": "element",
                }
            }
            output = reducer(loss_dict, None, None)
            self.assertTrue(output == loss_dict)
