import unittest

import torch
from torch.autograd import Variable

from pytorch_metric_learning.losses import PNPLoss

from .. import TEST_DEVICE, TEST_DTYPES
from ..zzz_testing_utils.testing_utils import angle_to_coord


class TestFastAPLoss(unittest.TestCase):
    def test_fast_ap_loss(self):
        b, alpha, anneal, variant = 2, 4, 0.01, "PNP-D_q"
        loss_func = PNPLoss(b, alpha, anneal, variant)
        ref_emb = torch.randn(32, 32)
        ref_labels = torch.randint(0, 10, (32,))

        for dtype in TEST_DTYPES:
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

            # pnp doesn't support ref_emb
            self.assertRaises(
                ValueError,
                lambda: loss_func(
                    embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels
                ),
            )
