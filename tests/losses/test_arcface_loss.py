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
            labels = torch.randint(low=0, high=10, size=(180,)).to(TEST_DEVICE)

            loss = loss_func(embeddings, labels)
            loss.backward()

            weights = F.normalize(loss_func.weight.detach().double(), p=2, dim=1).t()
            cosines = torch.matmul(F.normalize(embeddings.detach().double()), weights)
            logits = cosines.clone()

            for i, c in enumerate(labels):
                acos = torch.acos(torch.clamp(logits[i, c], -1, 1))
                if acos <= (np.pi - margin):
                    logits[i, c] = torch.cos(acos + margin)
                else:
                    logits[i, c] -= margin * np.sin(margin)

            correct_loss = F.cross_entropy(logits * scale, labels)
            helpers.assert_close(loss, correct_loss, dtype, atol=1e-8)

            logits_out = loss_func.get_logits(embeddings)
            helpers.assert_close(logits_out / scale, cosines, dtype)
