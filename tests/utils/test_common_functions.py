import unittest

import torch
from sklearn.preprocessing import StandardScaler

from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.utils import common_functions as c_f

from .. import WITH_COLLECT_STATS


class TestCommonFunctions(unittest.TestCase):
    def test_torch_standard_scaler(self):
        torch.manual_seed(56987)
        embeddings = torch.randn(1024, 512)
        scaled = c_f.torch_standard_scaler(embeddings)
        true_scaled = StandardScaler().fit_transform(embeddings.cpu().numpy())
        true_scaled = torch.from_numpy(true_scaled)
        self.assertTrue(torch.all(torch.isclose(scaled, true_scaled, rtol=1e-2)))

    def test_collect_stats_flag(self):
        self.assertTrue(c_f.COLLECT_STATS == WITH_COLLECT_STATS)
        loss_fn = TripletMarginLoss()
        self.assertTrue(loss_fn.collect_stats == WITH_COLLECT_STATS)
        self.assertTrue(loss_fn.distance.collect_stats == WITH_COLLECT_STATS)
        self.assertTrue(loss_fn.reducer.collect_stats == WITH_COLLECT_STATS)


if __name__ == "__main__":
    unittest.main()
