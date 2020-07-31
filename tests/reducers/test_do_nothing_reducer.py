import unittest
import torch
from pytorch_metric_learning.reducers import DoNothingReducer

class TestDoNothingReducer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.device = torch.device('cuda')

    def test_do_nothing_reducer(self):
        reducer = DoNothingReducer()
        for dtype in [torch.float16, torch.float32, torch.float64]:
            loss_dict = {"loss": {"losses": torch.randn(100).type(dtype).to(self.device), "indices": torch.arange(100), "reduction_type": "element"}}
            output = reducer(loss_dict, None, None)
            self.assertTrue(output == loss_dict)