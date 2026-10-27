import logging
import unittest

import torch

from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.utils import common_functions as c_f

from .. import WITH_COLLECT_STATS


class TestCommonFunctions(unittest.TestCase):
    def test_collect_stats_flag(self):
        self.assertTrue(c_f.COLLECT_STATS == WITH_COLLECT_STATS)
        for x in [True, False, True, False, WITH_COLLECT_STATS]:
            c_f.COLLECT_STATS = x
            loss_fn = TripletMarginLoss()
            self.assertTrue(loss_fn.collect_stats == x)
            self.assertTrue(loss_fn.distance.collect_stats == x)
            self.assertTrue(loss_fn.reducer.collect_stats == x)
        for x in [True, False]:
            loss_fn = TripletMarginLoss(collect_stats=x)
            self.assertTrue(loss_fn.collect_stats == x)

    def test_check_shapes(self):
        embeddings = torch.randn(32, 512, 3)
        labels = torch.randint(0, 10, size=(32,))
        loss_fn = TripletMarginLoss()

        # embeddings is 3-dimensional
        self.assertRaises(ValueError, lambda: loss_fn(embeddings, labels))

        # embeddings does not match labels
        embeddings = torch.randn(33, 512)
        self.assertRaises(ValueError, lambda: loss_fn(embeddings, labels))

        # labels is 2D
        embeddings = torch.randn(32, 512)
        labels = labels.unsqueeze(1)
        self.assertRaises(ValueError, lambda: loss_fn(embeddings, labels))

        # correct shapes
        labels = labels.squeeze(1)
        self.assertTrue(torch.is_tensor(loss_fn(embeddings, labels)))

    def test_logger(self):
        logging.basicConfig()
        for name in [None, "some_random_name"]:
            if name is not None:
                c_f.set_logger_name(name)
                self.assertTrue(c_f.LOGGER_NAME == name)
            for level in range(0, 60, 10):
                logging.getLogger(c_f.LOGGER_NAME).setLevel(level)
                self.assertTrue(c_f.LOGGER.level == level)
                self.assertTrue(logging.getLogger().level == logging.WARNING)


if __name__ == "__main__":
    unittest.main()
