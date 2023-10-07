import unittest

import torch
import numpy as np
import random

from pytorch_metric_learning.losses import (
    MultiSupConLoss,
    CrossBatchMemory4MultiLabel
)

class TestMultiSupConLoss(unittest.TestCase):
    def test_multi_supcon_loss(self):
        n_cls = 10
        n_samples = 16
        n_dim = 256
        loss_func = MultiSupConLoss(num_classes=10)
        xbm_loss_func = CrossBatchMemory4MultiLabel(loss_func, n_dim, memory_size=128)

        # # test float32 and float64
        # for dtype in [torch.float32, torch.float64]:
        #     embeddings = torch.randn(n_samples, n_dim, dtype=dtype)
        #     labels = [random.sample(range(n_cls), np.random.randint(1, 4)) for i in range(n_samples)]
        #     loss = loss_func(embeddings, labels)
        #     self.assertTrue(loss >= 0)

        # # test cuda and cpu
        # for device in [torch.device("cpu"),torch.device("cuda")]:
        #     embeddings = torch.randn(n_samples, n_dim, dtype=dtype, device=device)
        #     labels = [random.sample(range(n_cls), np.random.randint(1, 4)) for i in range(n_samples)]
        #     loss = loss_func(embeddings, labels)
        #     self.assertTrue(loss >= 0)

        # test xbm
        batchs = 10
        for b in range(batchs):
            embeddings = torch.randn(n_samples, n_dim, dtype=torch.float32)
            labels = [random.sample(range(n_cls), np.random.randint(1, 4)) for i in range(n_samples)]
            loss = xbm_loss_func(embeddings, labels)
            self.assertTrue(loss >= 0)