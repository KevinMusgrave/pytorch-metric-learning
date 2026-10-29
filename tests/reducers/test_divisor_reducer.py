import unittest

import torch

from pytorch_metric_learning.losses import ProxyAnchorLoss
from pytorch_metric_learning.reducers import (
    AvgNonZeroReducer,
    DivisorReducer,
    MeanReducer,
    MultipleReducers,
    SumReducer,
)

from .. import TEST_DEVICE, TEST_DTYPES


class TestDivisorReducer(unittest.TestCase):
    def test_divisor_reducer(self):
        reducer = DivisorReducer()
        batch_size = 100
        embedding_size = 64
        for dtype in TEST_DTYPES:
            embeddings = (
                torch.randn(batch_size, embedding_size).type(dtype).to(TEST_DEVICE)
            )
            labels = torch.randint(0, 10, (batch_size,))
            pair_indices = (
                torch.randint(0, batch_size, (batch_size,)),
                torch.randint(0, batch_size, (batch_size,)),
            )
            triplet_indices = (
                *pair_indices,
                torch.randint(0, batch_size, (batch_size,)),
            )
            losses = torch.randn(batch_size).type(dtype).to(TEST_DEVICE)

            for indices, reduction_type in [
                (torch.arange(batch_size), "element"),
                (pair_indices, "pos_pair"),
                (pair_indices, "neg_pair"),
                (triplet_indices, "triplet"),
                (None, "already_reduced"),
            ]:
                for partA, partB in [(0, 0), (32, 15)]:
                    L = losses[0] if reduction_type == "already_reduced" else losses
                    loss_dict = {
                        "loss": {
                            "losses": L,
                            "indices": indices,
                            "reduction_type": reduction_type,
                            "divisor": partA + partB,
                        }
                    }
                    output = reducer(loss_dict, embeddings, labels)
                    if reduction_type == "already_reduced":
                        correct_output = L
                    elif partA + partB == 0:
                        correct_output = torch.sum(L) * 0
                    else:
                        correct_output = torch.sum(L) / (32 + 15)
                    torch.testing.assert_close(output, correct_output.to(output.dtype))

            loss_dict = {
                "loss": {
                    "losses": losses[0],
                    "indices": None,
                    "reduction_type": "already_reduced",
                }
            }
            output = reducer(loss_dict, embeddings, labels)
            correct_output = losses[0]
            torch.testing.assert_close(output, correct_output.to(output.dtype))


class TestReducerTermContracts(unittest.TestCase):
    def test_divisors_require_divisor_reducer(self):
        embeddings = torch.randn(4, 2, device=TEST_DEVICE)
        labels = torch.tensor([0, 1, 0, 1], device=TEST_DEVICE)
        for reducer in (MeanReducer(), SumReducer(), AvgNonZeroReducer()):
            loss = ProxyAnchorLoss(2, 2, reducer=reducer).to(TEST_DEVICE)
            with self.assertRaisesRegex(ValueError, "DivisorReducer"):
                loss(embeddings, labels)

    def test_empty_loss_dict_reduces_to_connected_zero(self):
        embeddings = torch.randn(3, 2, device=TEST_DEVICE, requires_grad=True)
        for reducer in (MeanReducer(), DivisorReducer(), MultipleReducers({})):
            output = reducer({}, embeddings, None)
            output.backward()
            self.assertEqual(output.item(), 0)
            self.assertIsNotNone(embeddings.grad)
            embeddings.grad = None
