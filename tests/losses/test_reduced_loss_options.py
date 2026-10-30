import unittest
from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import SumReducer

from .. import TEST_DEVICE

REDUCED_LOSSES: tuple[Callable[..., losses.BaseMetricLossFunction], ...] = (
    losses.DynamicSoftMarginLoss,
    losses.HistogramLoss,
    partial(losses.ManifoldLoss, 4, 3),
    partial(losses.P2SGradLoss, 3, 4),
    losses.PNPLoss,
    losses.ThresholdConsistentMarginLoss,
)


class TestReducedLossOptions(unittest.TestCase):
    def test_losses_returning_one_reduced_value_reject_reducer(self):
        embeddings = torch.randn(6, 4, device=TEST_DEVICE)
        labels = torch.tensor([0, 0, 1, 1, 2, 2], device=TEST_DEVICE)
        for build in REDUCED_LOSSES:
            with self.subTest(loss=build):
                with self.assertRaises(TypeError):
                    build(reducer=SumReducer())
                terms = build().to(TEST_DEVICE).loss_terms(embeddings, labels)
                self.assertEqual([term["indices"] for term in terms.values()], [None])

    def test_p2s_grad_loss_rejects_distance(self):
        options: dict[str, Any] = {"distance": CosineSimilarity()}
        with self.assertRaises(TypeError):
            losses.P2SGradLoss(3, 4, **options)
