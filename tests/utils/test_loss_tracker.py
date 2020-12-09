import unittest
from pytorch_metric_learning.utils.loss_tracker import LossTracker


class TestLossTracker(unittest.TestCase):
    def test_loss_tracker(self):
        LT = LossTracker(["a", "b", "c"])
        self.assertTrue(LT.losses == {"a": 0, "b": 0, "c": 0, "total_loss": 0})

        losses = LT.losses
        losses["a"] = 12
        losses["b"] = 0.1
        losses["c"] = 3
        self.assertTrue(LT.losses == {"a": 12, "b": 0.1, "c": 3, "total_loss": 0})

        LT.update({})
        self.assertTrue(LT.losses["total_loss"] == losses["total_loss"] == 15.1)
        for k in losses.keys():
            losses[k] = 0

        self.assertTrue(LT.losses == {"a": 0, "b": 0, "c": 0, "total_loss": 0})

        losses["a"] = 12
        losses["b"] = 0.1
        losses["c"] = 3
        LT.update({"a": 3})
        self.assertTrue(LT.losses["total_loss"] == losses["total_loss"] == 39.1)


if __name__ == "__main__":
    unittest.main()
