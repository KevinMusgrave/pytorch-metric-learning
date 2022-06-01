import unittest

import torch

from pytorch_metric_learning.losses import InstanceLoss

from .. import TEST_DEVICE, TEST_DTYPES
from ..zzz_testing_utils.testing_utils import angle_to_coord



class TestInstanceLoss(unittest.TestCase):
    def test_instance_loss(self):
        for dtype in TEST_DTYPES:
            la = 1 if dtype == torch.float16 else 20
            gamma = 1 if dtype == torch.float16 else 0.1
            for gamma in range(1, 256, 32):
                loss_func = InstanceLoss(
                    gamma=gamma,
                ).to(TEST_DEVICE)

                embedding_angles = torch.arange(0, 180)
                embeddings = torch.tensor(
                    [angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(
                    TEST_DEVICE
                )  # 2D embeddings
                labels = torch.randint(low=0, high=10, size=(180,)).to(TEST_DEVICE)
                
                loss = loss_func(embeddings, labels)
                loss.backward()

if __name__ == "__main__":
    unittest.main()
