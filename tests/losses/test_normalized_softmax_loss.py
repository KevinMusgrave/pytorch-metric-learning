import unittest

import torch

from pytorch_metric_learning.losses import NormalizedSoftmaxLoss
from pytorch_metric_learning.utils import common_functions as c_f

from .. import TEST_DEVICE, TEST_DTYPES


class TestNormalizedSoftmaxLoss(unittest.TestCase):
    def test_normalized_softmax_loss(self):
        temperature = 0.1
        for dtype in TEST_DTYPES:
            loss_func = NormalizedSoftmaxLoss(
                temperature=temperature, num_classes=10, embedding_size=2
            )
            embedding_angles = torch.arange(0, 180)
            embeddings = torch.tensor(
                [c_f.angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.randint(low=0, high=10, size=(180,)).to(TEST_DEVICE)

            loss = loss_func(embeddings, labels)
            loss.backward()

            weights = torch.nn.functional.normalize(loss_func.W, p=2, dim=0)
            logits = torch.matmul(embeddings, weights)
            correct_loss = torch.nn.functional.cross_entropy(
                logits / temperature, labels
            )
            rtol = 1e-2 if dtype == torch.float16 else 1e-5
            self.assertTrue(torch.isclose(loss, correct_loss, rtol=rtol))

            # test get_logits
            logits_out = loss_func.get_logits(embeddings)
            self.assertTrue(
                torch.allclose(
                    logits_out, torch.matmul(embeddings, weights) / temperature, rtol=1e-2
                )
            )
