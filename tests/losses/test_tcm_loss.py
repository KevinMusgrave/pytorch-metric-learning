import unittest

import torch
import torch.nn.functional as F

from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import (
    ContrastiveLoss,
    MultipleLosses,
    ThresholdConsistentMarginLoss,
)

from .. import TEST_DEVICE, TEST_DTYPES


class TestThresholdConsistentMarginLoss(unittest.TestCase):
    def test_tcm_loss(self):
        torch.manual_seed(3459)
        for dtype in TEST_DTYPES:
            loss_func = MultipleLosses(
                losses=[
                    ContrastiveLoss(
                        distance=CosineSimilarity(),
                        pos_margin=0.9,
                        neg_margin=0.4,
                    ),
                    ThresholdConsistentMarginLoss(),
                ]
            )
            embs = torch.tensor(
                [
                    [0.00, 1.00],
                    [0.43, 0.90],
                    [1.00, 0.00],
                    [0.50, 0.50],
                ],
                device=TEST_DEVICE,
                dtype=dtype,
                requires_grad=True,
            )
            labels = torch.tensor([0, 0, 1, 1])

            # Contrastive loss = 0.4866
            #
            # TCM loss part:
            # Only pair (2, 3) is taken into account for positive part
            # Positive part = 1 * ( 0.9 - 0.7071 ) / ( 1 ) = 0.1929
            #
            # Only pairs (1, 2) and (1, 3) are taken into account for negative part
            # Negative part = 1 * ( 0.7071 - 0.5 + 0.9429 - 0.5  ) / ( 2 ) = 0.325
            #
            # Sum of these losses -> 0.4866 + 0.518 = 1.0046
            correct_loss = torch.tensor(1.0045).to(dtype)

            with torch.no_grad():
                res = loss_func.forward(embs, labels)
                rtol = 1e-2 if dtype == torch.float16 else 1e-5
                atol = 1e-4
                self.assertTrue(torch.isclose(res, correct_loss, rtol=rtol, atol=atol))
