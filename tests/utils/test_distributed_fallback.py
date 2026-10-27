import unittest

import torch

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.utils.distributed import (
    DistributedLossWrapper,
    DistributedMinerWrapper,
)


class TestDistributedFallback(unittest.TestCase):
    def test_loss_without_process_group_preserves_value_and_gradients(self):
        inputs = torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.3, 0.0], [0.8, 0.0]])
        labels = torch.tensor([0, 0, 1, 1])
        direct = inputs.clone().requires_grad_()
        wrapped = inputs.clone().requires_grad_()
        loss = ContrastiveLoss(distance=LpDistance(normalize_embeddings=False))
        expected = loss(direct, labels)
        with self.assertWarnsRegex(UserWarning, "non-distributed setting"):
            actual = DistributedLossWrapper(loss)(wrapped, labels)
        expected.backward()
        actual.backward()
        assert wrapped.grad is not None
        assert direct.grad is not None
        self.assertGreater(actual.item(), 0)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(wrapped.grad, direct.grad)
        self.assertGreater(wrapped.grad.abs().sum().item(), 0)

    def test_invalid_wrapped_objects_are_rejected(self):
        with self.assertRaises(TypeError):
            DistributedLossWrapper(torch.nn.MSELoss())
        with self.assertRaises(TypeError):
            DistributedMinerWrapper(torch.nn.Identity())
