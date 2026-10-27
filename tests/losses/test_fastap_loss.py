import unittest

import torch

from pytorch_metric_learning.losses import FastAPLoss

from .. import TEST_DEVICE, TEST_DTYPES
from ..zzz_testing_utils.testing_utils import angle_to_coord


def softBinning(D, mid, Delta):
    y = 1 - torch.abs(D - mid) / Delta
    return torch.max(torch.tensor([0], dtype=D.dtype).to(D.device), y)


def original_implementation_fast_ap_loss(input, target, num_bins):
    N = target.size()[0]
    assert input.size()[0] == N, "Batch size donesn't match!"

    Y = target.unsqueeze(1)
    Aff = 2 * (Y.t() == Y).type(input.dtype) - 1
    Aff.masked_fill_(torch.eye(N, N).bool().to(input.device), 0)

    I_pos = (Aff > 0).type(input.dtype).to(input.device)
    I_neg = (Aff < 0).type(input.dtype).to(input.device)
    N_pos = torch.sum(I_pos, 1)

    dist2 = 2 - 2 * torch.mm(input, input.t())

    Delta = torch.tensor(4.0 / num_bins).to(input.device)
    Z = torch.linspace(0.0, 4.0, steps=num_bins + 1).to(input.device)
    L = Z.size()[0]
    h_pos = torch.zeros((N, L), dtype=input.dtype).to(input.device)
    h_neg = torch.zeros((N, L), dtype=input.dtype).to(input.device)
    for bin_idx in range(L):
        pulse = softBinning(dist2, Z[bin_idx], Delta)
        h_pos[:, bin_idx] = torch.sum(pulse * I_pos, 1)
        h_neg[:, bin_idx] = torch.sum(pulse * I_neg, 1)

    H_pos = torch.cumsum(h_pos, 1)
    h = h_pos + h_neg
    H = torch.cumsum(h, 1)

    FastAP = h_pos * H_pos / H
    FastAP[torch.isnan(FastAP) | torch.isinf(FastAP)] = 0
    FastAP = torch.sum(FastAP, 1) / N_pos
    FastAP = FastAP[~torch.isnan(FastAP)]
    loss = 1 - torch.mean(FastAP)
    return loss


class TestFastAPLoss(unittest.TestCase):
    def test_fast_ap_loss(self):
        num_bins = 5
        loss_func = FastAPLoss(num_bins)
        ref_emb = torch.randn(32, 32)
        ref_labels = torch.randint(0, 10, (32,))

        for dtype in TEST_DTYPES:
            embedding_angles = torch.arange(0, 180)
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.randint(low=0, high=10, size=(180,)).to(TEST_DEVICE)

            loss = loss_func(embeddings, labels)
            loss.backward()
            original_loss = original_implementation_fast_ap_loss(
                torch.nn.functional.normalize(embeddings), labels, num_bins
            )
            rtol = 1e-2 if dtype == torch.float16 else 1e-5
            self.assertTrue(torch.isclose(loss, original_loss, rtol=rtol))

            with self.assertRaises(ValueError):
                loss_func(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
