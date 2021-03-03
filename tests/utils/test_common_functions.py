import unittest

import torch
from sklearn.preprocessing import StandardScaler

from pytorch_metric_learning.utils import common_functions as c_f


class TestCommonFunctions(unittest.TestCase):
    def test_torch_standard_scaler(self):
        torch.manual_seed(56987)
        embeddings = torch.randn(1024, 512)
        scaled = c_f.torch_standard_scaler(embeddings)
        true_scaled = StandardScaler().fit_transform(embeddings.cpu().numpy())
        true_scaled = torch.from_numpy(true_scaled)
        self.assertTrue(torch.all(torch.isclose(scaled, true_scaled, rtol=1e-2)))


if __name__ == "__main__":
    unittest.main()
