import unittest
from typing import Any

import torch
import torch.nn
import torch.nn.functional
import torch.nn.functional as F

from pytorch_metric_learning.losses import PNPLoss

from .. import TEST_DEVICE, TEST_DTYPES


class OriginalImplementationPNP(torch.nn.Module):
    def __init__(self, b, alpha, anneal, variant, bs, classes):
        super().__init__()
        self.b = b
        self.alpha = alpha
        self.anneal = anneal
        self.variant = variant
        self.batch_size = bs
        self.num_id = classes
        self.samples_per_class = int(bs / classes)

        mask = 1.0 - torch.eye(self.batch_size)
        for i in range(self.num_id):
            mask[
                i * (self.samples_per_class) : (i + 1) * (self.samples_per_class),
                i * (self.samples_per_class) : (i + 1) * (self.samples_per_class),
            ] = 0

        self.mask = mask.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)

    def forward(self, batch):
        dtype, device = batch.dtype, batch.device
        self.mask = self.mask.type(dtype).to(device)

        sim_all = self.compute_aff(batch)

        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        sim_sg = self.sigmoid(sim_diff, temp=self.anneal) * self.mask
        sim_all_rk = torch.sum(sim_sg, dim=-1)
        if self.variant == "PNP-D_s":
            sim_all_rk = torch.log(1 + sim_all_rk)
        elif self.variant == "PNP-D_q":
            sim_all_rk = 1 / (1 + sim_all_rk) ** (self.alpha)

        elif self.variant == "PNP-I_u":
            sim_all_rk = (1 + sim_all_rk) * torch.log(1 + sim_all_rk)

        elif self.variant == "PNP-I_b":
            b = self.b
            sim_all_rk = 1 / b**2 * (b * sim_all_rk - torch.log(1 + b * sim_all_rk))
        elif self.variant == "PNP-O":
            pass
        else:
            raise ValueError(f"variantation <{self.variant}> not available!")

        loss = torch.zeros(1).type(dtype).to(device)
        group = int(self.batch_size / self.num_id)

        for ind in range(self.num_id):
            neg_divide = torch.sum(
                sim_all_rk[
                    (ind * group) : ((ind + 1) * group),
                    (ind * group) : ((ind + 1) * group),
                ]
                / group
            )
            loss = loss + (neg_divide / self.batch_size)
        if self.variant == "PNP-D_q":
            return 1 - loss
        else:
            return loss

    def sigmoid(self, tensor, temp=1.0):
        exponent = -tensor / temp
        exponent = torch.clamp(exponent, min=-50, max=50)
        y = 1.0 / (1.0 + torch.exp(exponent))
        return y

    def compute_aff(self, x):
        return torch.mm(x, x.t())


class TestPNPLoss(unittest.TestCase):
    def test_pnp_loss(self):
        torch.manual_seed(30293)
        bs = 180
        classes = 30
        for variant in PNPLoss.VARIANTS:
            original_variant = {
                "Ds": "PNP-D_s",
                "Dq": "PNP-D_q",
                "Iu": "PNP-I_u",
                "Ib": "PNP-I_b",
                "O": "PNP-O",
            }[variant]
            b, alpha, anneal = 2, 4, 0.01
            loss_func = PNPLoss(b, alpha, anneal, variant)
            original_loss_func = OriginalImplementationPNP(
                b, alpha, anneal, original_variant, bs, classes
            ).to(TEST_DEVICE)

            for dtype in TEST_DTYPES:
                embeddings = torch.randn(
                    180, 32, dtype=dtype, device=TEST_DEVICE, requires_grad=True
                )
                labels = (
                    torch.tensor([[i] * (int(bs / classes)) for i in range(classes)])
                    .reshape(-1)
                    .to(TEST_DEVICE)
                )
                loss = loss_func(embeddings, labels)
                loss.backward()
                correct_loss = original_loss_func(F.normalize(embeddings, dim=-1))

                rtol = 1e-2 if dtype == torch.float16 else 1e-5
                self.assertTrue(torch.isclose(loss, correct_loss[0], rtol=rtol))

        invalid: Any = "PNP"
        with self.assertRaises(ValueError):
            PNPLoss(b, alpha, anneal, invalid)

    def test_negatives_that_have_no_positives(self):
        loss_func = PNPLoss()
        labels = torch.tensor([1, 1, 2], device=TEST_DEVICE)
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(3, 32, dtype=dtype, device=TEST_DEVICE)
            loss = loss_func(embeddings, labels)
            self.assertTrue(not torch.isnan(loss))
