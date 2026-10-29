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

            with self.assertRaisesRegex(ValueError, "divisor"):
                reducer(
                    {"loss": {"losses": losses[0], "indices": None, "divisor": 2}},
                    embeddings,
                    labels,
                )
            for indices in [torch.arange(batch_size), pair_indices, triplet_indices]:
                for partA, partB in [(0, 0), (32, 15)]:
                    L = losses
                    loss_dict = {
                        "loss": {
                            "losses": L,
                            "indices": indices,
                            "divisor": partA + partB,
                        }
                    }
                    output = reducer(loss_dict, embeddings, labels)
                    if partA + partB == 0:
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


class TestDivisorPrecision(unittest.TestCase):
    def test_divisor_is_built_in_the_accumulation_dtype(self):
        cases = [
            (torch.float64, [3e-50, 1e-50], 1e-50, 4.0),
            (torch.float64, [2.0, 4.0], torch.tensor(1e-50, dtype=torch.float64), 6e50),
            (torch.float32, [1e-30, 1e-30], 1e-30, 2.0),
            (torch.float16, [3.0, 5.0], 1e-3, 8e3),
        ]
        for dtype, values, divisor, expected in cases:
            with self.subTest(dtype=dtype, divisor=divisor):
                losses = torch.tensor(
                    values, dtype=dtype, device=TEST_DEVICE, requires_grad=True
                )
                if isinstance(divisor, torch.Tensor):
                    divisor = divisor.to(TEST_DEVICE)
                output = DivisorReducer()(
                    {
                        "loss": {
                            "losses": losses,
                            "indices": torch.arange(2, device=TEST_DEVICE),
                            "divisor": divisor,
                        }
                    },
                    torch.zeros(2, 1, dtype=dtype, device=TEST_DEVICE),
                    None,
                )
                (gradient,) = torch.autograd.grad(output, losses)
                exact_divisor = float(divisor)
                torch.testing.assert_close(
                    output.double().cpu(),
                    torch.tensor(expected, dtype=torch.float64),
                    rtol=1e-3 if dtype == torch.float16 else 1e-12,
                    atol=0,
                )
                torch.testing.assert_close(
                    gradient.double().cpu(),
                    torch.full((2,), 1 / exact_divisor, dtype=torch.float64),
                    rtol=1e-3 if dtype == torch.float16 else 1e-6,
                    atol=0,
                )
