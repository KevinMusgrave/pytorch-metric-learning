import unittest

import torch

from pytorch_metric_learning.distances import DotProductSimilarity
from pytorch_metric_learning.losses import NormalizedSoftmaxLoss

from .. import TEST_DEVICE, TEST_DTYPES, helpers
from ..helpers import angle_to_coord


class TestNormalizedSoftmaxLoss(unittest.TestCase):
    def test_normalized_softmax_loss(self):
        temperature = 0.1
        for dtype in TEST_DTYPES:
            loss_func = NormalizedSoftmaxLoss(
                temperature=temperature,
                num_classes=10,
                embedding_size=2,
                distance=DotProductSimilarity(normalize_embeddings=False),
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

            weights = torch.nn.functional.normalize(
                loss_func.weight.detach().double(), p=2, dim=1
            ).t()
            similarities = torch.matmul(embeddings.detach().double(), weights)
            correct_loss = torch.nn.functional.cross_entropy(
                similarities / temperature, labels
            )
            helpers.assert_close(loss, correct_loss, dtype, atol=1e-8)

            logits_out = loss_func.get_logits(embeddings)
            helpers.assert_close(logits_out * temperature, similarities, dtype)
