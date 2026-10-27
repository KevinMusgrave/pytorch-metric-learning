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
from pytorch_metric_learning.losses import ThresholdConsistentMarginLoss


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


class TestThresholdConsistentMarginReference(unittest.TestCase):
    def test_reference_embeddings_are_used(self):
        torch.manual_seed(7)
        query = torch.randn(4, 3, dtype=torch.float64)
        reference = torch.randn(6, 3, dtype=torch.float64)
        labels = torch.tensor([0, 1, 0, 1])
        ref_labels = torch.tensor([0, 0, 1, 1, 2, 1])
        loss = ThresholdConsistentMarginLoss(margin_plus=0.5, margin_minus=-0.5)(
            query, labels, ref_emb=reference, ref_labels=ref_labels
        )
        sims = F.normalize(query, dim=1) @ F.normalize(reference, dim=1).T
        same = labels[:, None] == ref_labels[None, :]
        pos, neg = sims[same], sims[~same]
        pos_terms = (0.5 - pos)[pos <= 0.5]
        neg_terms = (neg + 0.5)[neg >= -0.5]
        expected = pos_terms.mean() + neg_terms.mean()
        torch.testing.assert_close(loss, expected)
