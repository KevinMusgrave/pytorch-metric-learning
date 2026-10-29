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

    def mine(self, embeddings, labels, ref_emb, ref_labels):
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
        for reduction, indices in [
            ("element", index),
            ("pos_pair", (index, index)),
            ("neg_pair", (index, index)),
            ("triplet", (index, index, index)),
        ]:
            with self.subTest(reduction=reduction):
                term = {"indices": indices, "reduction_type": reduction}
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
            ((index, index[:1], index), ValueError, "equal length"),
            ((index, index, index, index[:1]), ValueError, "equal length"),
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
            (values, index, "unknown", ValueError, "Unknown reduction_type"),
            (values, index[:1], "element", ValueError, "equal lengths"),
            (values, (index, index[:1]), "pos_pair", ValueError, "equal lengths"),
            (values, (index, index, [0, 1]), "triplet", TypeError, "tensor"),
            (values, None, "already_reduced", ValueError, "one value"),
            (values, index[:, None], "element", ValueError, "one-dimensional"),
        ]
        for losses, indices, reduction, error, message in cases:
            with (
                self.subTest(reduction=reduction, indices=indices),
                self.assertRaisesRegex(error, message),
            ):
                MeanReducer()(
                    {
                        "loss": {
                            "losses": losses,
                            "indices": indices,
                            "reduction_type": reduction,
                        }
                    },
                    torch.randn(2, 3),
                    torch.arange(2),
                )

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
