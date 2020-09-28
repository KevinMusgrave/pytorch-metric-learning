import unittest
from .. import TEST_DTYPES, TEST_DEVICE
import torch
import numpy as np
from pytorch_metric_learning.losses import ArcFaceLoss
from pytorch_metric_learning.utils import common_functions as c_f


class TestArcFaceLoss(unittest.TestCase):
    def test_arcface_loss(self):
        margin = 30
        scale = 64
        for dtype in TEST_DTYPES:
            loss_func = ArcFaceLoss(
                margin=margin, scale=scale, num_classes=10, embedding_size=2
            )
            embedding_angles = torch.arange(0, 180)
            embeddings = torch.tensor(
                [c_f.angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.randint(low=0, high=10, size=(180,))

            loss = loss_func(embeddings, labels)
            loss.backward()

            weights = torch.nn.functional.normalize(loss_func.W, p=2, dim=0)
            logits = torch.matmul(embeddings, weights)

            for i, c in enumerate(labels):
                acos = torch.acos(torch.clamp(logits[i, c], -1, 1))
                logits[i, c] = torch.cos(
                    acos + torch.tensor(np.radians(margin), dtype=dtype).to(TEST_DEVICE)
                )

            correct_loss = torch.nn.functional.cross_entropy(
                logits * scale, labels.to(TEST_DEVICE)
            )

            rtol = 1e-2 if dtype == torch.float16 else 1e-5
            self.assertTrue(torch.isclose(loss, correct_loss, rtol=rtol))
