import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_metric_learning.losses import ArcFaceLoss

from .. import TEST_DEVICE, TEST_DTYPES, helpers
from ..helpers import angle_to_coord


class TestArcFaceLoss(unittest.TestCase):
    def test_arcface_loss(self):
        margin = np.radians(30)
        scale = 64
        for dtype in TEST_DTYPES:
            loss_func = ArcFaceLoss(
                margin=margin, scale=scale, num_classes=10, embedding_size=2
            ).to(TEST_DEVICE, dtype)
            embedding_angles = torch.arange(0, 180)
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.randint(low=0, high=10, size=(180,))

            loss = loss_func(embeddings, labels)
            loss.backward()

            weights = F.normalize(loss_func.weight, p=2, dim=1).t()
            logits = torch.matmul(F.normalize(embeddings), weights)

            for i, c in enumerate(labels):
                acos = torch.acos(torch.clamp(logits[i, c], -1, 1))
                if acos <= (np.pi - margin):
                    logits[i, c] = torch.cos(
                        acos + torch.tensor(margin, dtype=dtype).to(TEST_DEVICE)
                    )
                else:
                    mg = margin
                    logits[i, c] -= torch.tensor(mg * np.sin(mg), dtype=dtype).to(
                        TEST_DEVICE
                    )

            correct_loss = F.cross_entropy(logits * scale, labels.to(TEST_DEVICE))

            rtol = helpers.rtol(dtype)
            self.assertTrue(torch.isclose(loss, correct_loss, rtol=rtol))

            logits_out = loss_func.get_logits(embeddings)
            self.assertTrue(
                torch.allclose(
                    logits_out, torch.matmul(F.normalize(embeddings), weights) * scale
                )
            )
