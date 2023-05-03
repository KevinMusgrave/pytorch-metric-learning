import unittest

import torch

from pytorch_metric_learning.distances import LpDistance

from .. import WITH_COLLECT_STATS


class TestCollectedStats(unittest.TestCase):
    @unittest.skipUnless(WITH_COLLECT_STATS, "WITH_COLLECT_STATS is false")
    def test_collected_stats(self):
        x = torch.randn(32, 128)
        d = LpDistance()
        d(x)

        self.assertNotEqual(d.initial_avg_query_norm, 0)
        self.assertNotEqual(d.initial_avg_ref_norm, 0)
        self.assertNotEqual(d.final_avg_query_norm, 0)
        self.assertNotEqual(d.final_avg_ref_norm, 0)
