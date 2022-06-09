import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pytorch_metric_learning.losses import InstanceLoss

from .. import TEST_DEVICE, TEST_DTYPES


# original loss copied from here
# https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/instance_loss.py
# as a reference to test against
def l2_norm(v):
    fnorm = torch.norm(v, p=2, dim=1, keepdim=True) + 1e-6
    v = v.div(fnorm.expand_as(v))
    return v


class OriginalInstanceLoss(nn.Module):
    def __init__(self, gamma=1) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, feature, label=None) -> Tensor:
        # Dual-Path Convolutional Image-Text Embeddings with Instance Loss, ACM TOMM 2020
        # https://zdzheng.xyz/files/TOMM20.pdf
        # using cross-entropy loss for every sample if label is not available. else use given label.
        normed_feature = l2_norm(feature)
        sim1 = torch.mm(normed_feature * self.gamma, torch.t(normed_feature))
        # sim2 = sim1.t()
        if label is None:
            sim_label = torch.arange(sim1.size(0)).cuda().detach()
        else:
            _, sim_label = torch.unique(label, return_inverse=True)
        loss = F.cross_entropy(sim1, sim_label)  # + F.cross_entropy(sim2, sim_label)
        return loss


class TestInstanceLoss(unittest.TestCase):
    def test_instance_loss(self):
        torch.manual_seed(3029)
        for dtype in TEST_DTYPES:
            for gamma in range(1, 256, 32):
                loss_fn = InstanceLoss(gamma=gamma)
                original_loss_fn = OriginalInstanceLoss(gamma=gamma)
                embeddings = torch.randn(
                    32, 128, device=TEST_DEVICE, dtype=dtype, requires_grad=True
                )
                labels = torch.randint(0, 10, size=(32,), device=TEST_DEVICE)
                loss = loss_fn(embeddings, labels)
                loss.backward()

                original_loss = original_loss_fn(embeddings, labels)
                rtol = 1e-3 if dtype == torch.float16 else 1e-6
                self.assertTrue(
                    np.isclose(loss.item(), original_loss.item(), rtol=rtol)
                )


if __name__ == "__main__":
    unittest.main()
