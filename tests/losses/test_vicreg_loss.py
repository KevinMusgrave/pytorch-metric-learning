import unittest

import torch
import torch.nn.functional as F

from pytorch_metric_learning.losses import VICRegLoss

from .. import TEST_DEVICE, TEST_DTYPES
from ..zzz_testing_utils.testing_utils import angle_to_coord

HYPERPARAMETERS = [[25, 25, 1, 1e-4], [10, 10, 2, 1e-5], [5, 5, 5, 1e-6]]


class TestVICRegLoss(unittest.TestCase):
    def test_vicreg_loss(self):
        torch.manual_seed(3459)
        for dtype in TEST_DTYPES:
            for hyp in HYPERPARAMETERS:
                loss_func = VICRegLoss(
                    invariance_lambda=hyp[0],
                    variance_mu=hyp[1],
                    covariance_v=hyp[2],
                    eps=hyp[3],
                )
                ref_emb_ = torch.randn(
                    32, 64, device=TEST_DEVICE, dtype=dtype, requires_grad=True
                )
                augmentation_noise = torch.normal(
                    0, 0.1, size=(32, 64), device=TEST_DEVICE, dtype=dtype
                )
                emb_ = ref_emb_ + augmentation_noise

                for emb, ref_emb in [(emb_, ref_emb_), (ref_emb_, emb_)]:
                    loss = loss_func(ref_emb, emb)
                    loss.backward()

                    # invariance_loss
                    invariance_loss = F.mse_loss(emb, ref_emb)

                    # variance_loss
                    std_emb = torch.sqrt(emb.var(dim=0) + hyp[3])
                    std_ref_emb = torch.sqrt(ref_emb.var(dim=0) + hyp[3])
                    variance_loss = torch.mean(F.relu(1 - std_emb)) + torch.mean(
                        F.relu(1 - std_ref_emb)
                    )

                    # covariance loss, a more manual version
                    N, D = emb.size()
                    emb = emb - emb.mean(dim=0)
                    ref_emb = ref_emb - ref_emb.mean(dim=0)
                    cov_emb = (emb.T @ emb) / (N - 1)
                    cov_ref_emb = (ref_emb.T @ ref_emb) / (N - 1)
                    diag = torch.eye(D, device=emb.device)
                    covariance_loss = (
                        cov_emb[~diag.bool()].pow_(2).sum() / D
                        + cov_ref_emb[~diag.bool()].pow_(2).sum() / D
                    )

                    correct_loss = (
                        hyp[0] * invariance_loss
                        + hyp[1] * variance_loss
                        + hyp[2] * covariance_loss
                    )

                    rtol = 1e-2 if dtype == torch.float16 else 1e-5

                    self.assertTrue(torch.isclose(loss, correct_loss, rtol=rtol))
