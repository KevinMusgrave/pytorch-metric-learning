import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_metric_learning.losses import VICRegLoss

from .. import TEST_DEVICE, TEST_DTYPES
from ..zzz_testing_utils.testing_utils import angle_to_coord

HYPERPARAMETERS = [
    [25, 25, 1, 1e-4],
    [10, 10, 2, 1e-5],
    [5, 5, 5, 1e-6]
]
class TestVICRegLoss(unittest.TestCase):
    def test_vicreg_loss(self):
        for dtype in TEST_DTYPES:
            for hyp in HYPERPARAMETERS:
                loss_func = VICRegLoss(
                    invariance_lambda=hyp[0],
                    variance_mu=hyp[1],
                    covariance_v=hyp[2],
                    eps=hyp[3]
                )
                embedding_angles = torch.arange(0, 180)
                ref_emb = torch.tensor(
                    [angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(
                    TEST_DEVICE
                )  # 2D embeddings
                augmentation_noise = torch.normal(0, 0.1, size=(180,2), device=TEST_DEVICE, dtype=dtype)
                emb = ref_emb + augmentation_noise

                loss = loss_func(emb, ref_emb)
                loss.backward()

                # invariance_loss
                invariance_loss = F.mse_loss(emb, ref_emb)
                
                # variance_loss
                std_emb = torch.sqrt(emb.var(dim=0) + hyp[3])
                std_ref_emb = torch.sqrt(ref_emb.var(dim=0) + hyp[3])
                variance_loss = torch.mean(F.relu(1 - std_emb)) + torch.mean(F.relu(1 - std_ref_emb))

                # covariance loss, a more manual version
                N, D = emb.size()
                emb = emb - emb.mean(dim=0)
                ref_emb = ref_emb - ref_emb.mean(dim=0)
                cov_emb = (emb.T @ emb) / (N - 1)
                cov_ref_emb = (ref_emb.T @ ref_emb) / (N - 1)
                diag = torch.eye(D, device=emb.device)
                covariance_loss = cov_emb[~diag.bool()].pow_(2).sum() / D + cov_ref_emb[~diag.bool()].pow_(2).sum() / D
                
                correct_loss = hyp[0] * invariance_loss + hyp[1] * variance_loss + hyp[2] * covariance_loss

                rtol = 1e-2 if dtype == torch.float16 else 1e-5

                self.assertTrue(torch.isclose(loss, correct_loss, rtol=rtol))
    
    def test_order_invariance(self):
        for dtype in TEST_DTYPES:
            for hyp in HYPERPARAMETERS:
                loss_func = VICRegLoss(
                    invariance_lambda=hyp[0],
                    variance_mu=hyp[1],
                    covariance_v=hyp[2],
                    eps=hyp[3]
                )
                embedding_angles = torch.arange(0, 180)
                ref_emb = torch.tensor(
                    [angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(
                    TEST_DEVICE
                )  # 2D embeddings
                augmentation_noise = torch.normal(0, 0.1, size=(180,2), device=TEST_DEVICE, dtype=dtype)
                emb = ref_emb + augmentation_noise

                loss = loss_func(ref_emb, emb)
                loss.backward()

                # invariance_loss
                invariance_loss = F.mse_loss(emb, ref_emb)
                
                # variance_loss
                std_emb = torch.sqrt(emb.var(dim=0) + hyp[3])
                std_ref_emb = torch.sqrt(ref_emb.var(dim=0) + hyp[3])
                variance_loss = torch.mean(F.relu(1 - std_emb)) + torch.mean(F.relu(1 - std_ref_emb))

                # covariance loss, a more manual version
                N, D = emb.size()
                emb = emb - emb.mean(dim=0)
                ref_emb = ref_emb - ref_emb.mean(dim=0)
                cov_emb = (emb.T @ emb) / (N - 1)
                cov_ref_emb = (ref_emb.T @ ref_emb) / (N - 1)
                diag = torch.eye(D, device=emb.device)
                covariance_loss = cov_emb[~diag.bool()].pow_(2).sum() / D + cov_ref_emb[~diag.bool()].pow_(2).sum() / D
                
                correct_loss = hyp[0] * invariance_loss + hyp[1] * variance_loss + hyp[2] * covariance_loss

                rtol = 1e-2 if dtype == torch.float16 else 1e-5

                self.assertTrue(torch.isclose(loss, correct_loss, rtol=rtol))
    

                