import unittest
from typing import Any

import torch

from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.losses import (
    CircleLoss,
    ContrastiveLoss,
    CrossBatchMemory,
    MarginLoss,
    P2SGradLoss,
)
from pytorch_metric_learning.miners import BaseMiner
from pytorch_metric_learning.reducers import MeanReducer


class InvalidMiner(BaseMiner):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def mine(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        return self.output


class TestValidation(unittest.TestCase):
    def test_label_dependent_parameters_require_labels(self):
        embeddings = torch.eye(3)
        with self.assertRaisesRegex(ValueError, "labels are required"):
            P2SGradLoss(3, 3)(embeddings)
        indices = tuple(torch.tensor([i]) for i in range(3))
        with self.assertRaisesRegex(ValueError, "labels are required"):
            MarginLoss(num_classes=3)(embeddings, indices_tuple=indices)

    def test_reducer_requires_vector_losses(self):
        index = torch.arange(2)
        for kind, indices in [
            ("element", index),
            ("pair", (index, index)),
            ("triplet", (index, index, index)),
        ]:
            with self.subTest(kind=kind):
                term = {"indices": indices}
                with self.assertRaisesRegex(ValueError, "one-dimensional"):
                    MeanReducer()(
                        {"loss": {"losses": torch.tensor([[2.0], [4.0]]), **term}},
                        torch.randn(2, 3),
                        torch.arange(2),
                    )
                values = torch.tensor([2.0, 4.0], requires_grad=True)
                loss = MeanReducer()(
                    {"loss": {"losses": values, **term}}, torch.randn(2, 3), index
                )
                self.assertEqual(loss.item(), 3.0)
                gradient = torch.autograd.grad(loss, values)[0]
                torch.testing.assert_close(gradient, torch.full_like(values, 0.5))

    def test_miner_rejects_malformed_outputs(self):
        index = torch.arange(2)
        cases = [
            ((index, index), ValueError, "three triplet or four pair"),
            ((index, index[:1], index), ValueError, "equal-length"),
            ((index, index, index, index[:1]), ValueError, "equal-length"),
            ((index, index, [0, 1]), TypeError, "index tensors"),
            ((index, index, index[:, None]), ValueError, "one-dimensional"),
        ]
        for output, error, message in cases:
            with self.subTest(output=output), self.assertRaisesRegex(error, message):
                InvalidMiner(output)(torch.randn(2, 3), torch.arange(2))

    def test_reducer_rejects_malformed_loss_terms(self):
        values = torch.tensor([1.0, 2.0], requires_grad=True)
        index = torch.arange(2)
        cases = [
            (values, index[:1], ValueError, "equal lengths"),
            (values, (index, index[:1]), ValueError, "equal lengths"),
            (values, (index, index, [0, 1]), TypeError, "tensor"),
            (values, (index,) * 4, TypeError, "two or three tensors"),
            (values, (index,), TypeError, "two or three tensors"),
            (values, index.float(), TypeError, "int64 or int32"),
            (values, (index, index.bool()), TypeError, "int64 or int32"),
            (
                values,
                (index, index, index.to(torch.uint8)),
                TypeError,
                "int64 or int32",
            ),
            (values, None, ValueError, "one value"),
            (values, index[:, None], ValueError, "one-dimensional"),
        ]
        for losses, indices, error, message in cases:
            with (
                self.subTest(indices=indices),
                self.assertRaisesRegex(error, message),
            ):
                MeanReducer()(
                    {"loss": {"losses": losses, "indices": indices}},
                    torch.randn(2, 3),
                    torch.arange(2),
                )

    def test_reducer_accepts_int32_and_int64_indices(self):
        values = torch.tensor([1.0, 3.0], requires_grad=True)
        for dtype in (torch.int32, torch.int64):
            index = torch.arange(2, dtype=dtype)
            for indices in (index, (index, index), (index, index, index)):
                with self.subTest(dtype=dtype, kind=len(indices)):
                    loss = MeanReducer()(
                        {"loss": {"losses": values, "indices": indices}},
                        torch.randn(2, 3),
                        torch.arange(2),
                    )
                    (gradient,) = torch.autograd.grad(loss, values)
                    self.assertEqual(loss.item(), 2.0)
                    torch.testing.assert_close(gradient, torch.full((2,), 0.5))

    def test_indices_tuple_requires_integer_index_tensors(self):
        embeddings, labels = torch.randn(3, 2), torch.tensor([0, 0, 1])
        index = torch.tensor([0, 1])
        for bad in (index.float(), index.bool(), index.to(torch.uint8)):
            pairs = (index, index, index, bad)
            with self.subTest(dtype=bad.dtype):
                with self.assertRaisesRegex(TypeError, "int64 or int32"):
                    ContrastiveLoss()(embeddings, labels, pairs)
                with self.assertRaisesRegex(TypeError, "int64 or int32"):
                    InvalidMiner(pairs)(embeddings, labels)
                with self.assertRaisesRegex(TypeError, "int64 or int32"):
                    InvalidMiner(None)(embeddings, labels, pairs)
        small = torch.tensor([0, 1], dtype=torch.int32)
        expected = ContrastiveLoss()(
            embeddings, labels, (index, index + 1, index, index + 1)
        )
        actual = ContrastiveLoss()(
            embeddings, labels, (small, small + 1, small, small + 1)
        )
        torch.testing.assert_close(actual, expected)

    def test_distance_contracts(self):
        with self.assertRaisesRegex(TypeError, "CosineSimilarity"):
            CircleLoss(distance=LpDistance())
        unknown: dict[str, Any] = {"is_inverted": True}
        with self.assertRaisesRegex(TypeError, "is_inverted"):
            LpDistance(**unknown)
        points = torch.eye(2)
        self.assertTrue(torch.equal(CosineSimilarity()(points), points))

    def test_memory_rejects_invalid_mask_before_enqueueing(self):
        loss = CrossBatchMemory(ContrastiveLoss(), embedding_size=3, memory_size=4)
        embeddings, labels = torch.randn(2, 3), torch.arange(2)
        for mask, error, message in [
            (torch.tensor([0, 1]), TypeError, "boolean"),
            (torch.tensor([True]), ValueError, "one value per embedding"),
            (torch.tensor([[True, False]]), ValueError, "one-dimensional"),
        ]:
            with self.subTest(mask=mask):
                with self.assertRaisesRegex(error, message):
                    loss(embeddings, labels, enqueue_mask=mask)
                self.assertEqual(loss.queue_idx, 0)
                self.assertFalse(loss.has_been_filled)


if __name__ == "__main__":
    unittest.main()
