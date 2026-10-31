import unittest
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
import torch.distributed as dist

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.distributed import (
    DistributedLossWrapper,
    DistributedMinerWrapper,
)
from pytorch_metric_learning.losses import ContrastiveLoss, CrossBatchMemory
from pytorch_metric_learning.miners import PairMarginMiner
from pytorch_metric_learning.types import DistributedObjective


@contextmanager
def single_process_group(initialized: bool) -> Iterator[None]:
    if not initialized:
        yield
        return
    with TemporaryDirectory() as directory:
        dist.init_process_group(
            "gloo",
            init_method=(Path(directory) / "rendezvous").as_uri(),
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=15),
        )
        try:
            yield
        finally:
            dist.destroy_process_group()


def make_loss() -> ContrastiveLoss:
    return ContrastiveLoss(distance=LpDistance(normalize_embeddings=False))


class TestDistributedFallback(unittest.TestCase):
    def test_loss_without_process_group_preserves_value_and_gradients(self):
        inputs = torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.3, 0.0], [0.8, 0.0]])
        labels = torch.tensor([0, 0, 1, 1])
        direct = inputs.clone().requires_grad_()
        wrapped = inputs.clone().requires_grad_()
        loss = ContrastiveLoss(distance=LpDistance(normalize_embeddings=False))
        expected = loss(direct, labels)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            actual = DistributedLossWrapper(loss)(wrapped, labels)
        expected.backward()
        actual.backward()
        assert wrapped.grad is not None
        assert direct.grad is not None
        self.assertGreater(actual.item(), 0)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(wrapped.grad, direct.grad)
        self.assertGreater(wrapped.grad.abs().sum().item(), 0)

    def test_miner_preserves_pairs_with_and_without_references(self):
        inputs = torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.3, 0.0], [0.8, 0.0]])
        labels = torch.tensor([0, 0, 1, 1])
        reference = torch.tensor([[0.05, 0.0], [0.6, 0.0]])
        reference_labels = torch.tensor([0, 1])
        miner = PairMarginMiner(
            pos_margin=0.1,
            neg_margin=0.35,
            distance=LpDistance(normalize_embeddings=False),
        )
        for initialized in (False, True):
            with (
                self.subTest(initialized=initialized),
                single_process_group(initialized),
            ):
                for objective in ("global", "local_anchor"):
                    wrapped = DistributedMinerWrapper(miner, objective=objective)
                    for refs in ((None, None), (reference, reference_labels)):
                        expected = miner(inputs, labels, None, *refs)
                        actual = wrapped(inputs, labels, None, *refs)
                        self.assertGreater(len(actual[0]), 0)
                        self.assertGreater(len(actual[2]), 0)
                        for observed, wanted in zip(actual, expected, strict=True):
                            torch.testing.assert_close(observed, wanted)

    def test_loss_preserves_reference_gradients_and_supplied_pairs(self):
        labels = torch.tensor([0, 1])
        ref_labels = torch.tensor([0, 0, 1, 1])
        pairs = tuple(torch.tensor(x) for x in ([0, 1], [1, 3], [0, 1], [2, 0]))
        for initialized in (False, True):
            with (
                self.subTest(initialized=initialized),
                single_process_group(initialized),
            ):
                query = torch.tensor([[0.1, 0.2], [0.6, 0.3]], requires_grad=True)
                refs = torch.tensor(
                    [[0.0, 0.0], [0.2, 0.1], [0.3, 0.0], [0.8, 0.2]],
                    requires_grad=True,
                )
                loss = make_loss()
                expected = loss(query, labels, pairs, refs, ref_labels)
                expected_grads = torch.autograd.grad(expected, (query, refs))
                actual = DistributedLossWrapper(loss)(
                    query, labels, pairs, refs, ref_labels
                )
                actual_grads = torch.autograd.grad(actual, (query, refs))
                torch.testing.assert_close(actual, expected)
                for observed, wanted in zip(actual_grads, expected_grads, strict=True):
                    torch.testing.assert_close(observed, wanted)
                    self.assertGreater(observed.abs().sum().item(), 0)

    def test_memory_preserves_queue_and_gradients(self):
        for initialized in (False, True):
            with (
                self.subTest(initialized=initialized),
                single_process_group(initialized),
            ):
                for mask in (None, torch.tensor([True, False, False, True])):
                    for objective in ("global", "local_anchor"):
                        with self.subTest(mask=mask, objective=objective):
                            self.check_memory_sequence(mask, objective)

    def check_memory_sequence(
        self, mask: torch.Tensor | None, objective: DistributedObjective
    ) -> None:
        direct = CrossBatchMemory(make_loss(), embedding_size=2, memory_size=5)
        wrapped = DistributedLossWrapper(
            CrossBatchMemory(make_loss(), embedding_size=2, memory_size=5),
            objective=objective,
        )
        expected_memory = torch.zeros(5, 2)
        expected_labels = torch.zeros(5, dtype=torch.long)
        labels = torch.tensor([0, 0, 1, 1])
        queued = 0
        for step in range(3):
            inputs = torch.tensor([[0.0, 0.0], [0.2, 0.1], [0.3, -0.1], [0.8, 0.2]])
            inputs = (inputs + step * 0.1).requires_grad_()
            expected = direct(inputs, labels, enqueue_mask=mask)
            expected_grad = torch.autograd.grad(expected, inputs)[0]
            actual = wrapped(inputs, labels, enqueue_mask=mask)
            actual_grad = torch.autograd.grad(actual, inputs)[0]
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual_grad, expected_grad)
            self.assertGreater(actual_grad.abs().sum().item(), 0)
            queue_rows = torch.ones(4, dtype=torch.bool) if mask is None else mask
            count = int(queue_rows.sum())
            positions = torch.arange(queued, queued + count) % 5
            expected_memory[positions] = inputs.detach()[queue_rows]
            expected_labels[positions] = labels[queue_rows]
            queued += count
            torch.testing.assert_close(wrapped.loss.embedding_memory, expected_memory)
            torch.testing.assert_close(wrapped.loss.label_memory, expected_labels)
            self.assertEqual(wrapped.loss.queue_idx, queued % 5)
            self.assertEqual(wrapped.loss.has_been_filled, queued >= 5)
            self.assertFalse(wrapped.loss.embedding_memory.requires_grad)
            if mask is not None:
                torch.testing.assert_close(
                    actual_grad[mask], torch.zeros_like(actual_grad[mask])
                )

    def test_memory_rejects_reference_inputs_before_queue_updates(self):
        inputs = torch.tensor([[0.0, 0.0], [0.2, 0.1]])
        labels = torch.tensor([0, 0])
        for initialized in (False, True):
            with (
                self.subTest(initialized=initialized),
                single_process_group(initialized),
            ):
                loss = CrossBatchMemory(make_loss(), embedding_size=2, memory_size=5)
                wrapper = DistributedLossWrapper(loss)
                for reference in ({"ref_emb": inputs}, {"ref_labels": labels}):
                    with self.assertRaisesRegex(
                        ValueError, "not compatible with ref_emb"
                    ):
                        wrapper(inputs, labels, **reference)
                    self.assertEqual(loss.queue_idx, 0)
                    torch.testing.assert_close(loss.embedding_memory, torch.zeros(5, 2))

    def test_enqueue_mask_requires_memory(self):
        inputs = torch.tensor([[0.0, 0.0], [0.2, 0.1]])
        labels = torch.tensor([0, 0])
        for initialized in (False, True):
            with (
                self.subTest(initialized=initialized),
                single_process_group(initialized),
                self.assertRaisesRegex(ValueError, "enqueue_mask.*CrossBatchMemory"),
            ):
                DistributedLossWrapper(make_loss())(
                    inputs, labels, enqueue_mask=torch.tensor([True, False])
                )

    def test_invalid_wrapped_objects_are_rejected(self):
        invalid: Any = torch.nn.Identity()
        with self.assertRaises(TypeError):
            DistributedLossWrapper(invalid)
        with self.assertRaises(TypeError):
            DistributedMinerWrapper(invalid)

    def test_invalid_objective_is_rejected(self):
        objective: Any = "local"
        with self.assertRaisesRegex(ValueError, "objective"):
            DistributedLossWrapper(ContrastiveLoss(), objective=objective)
        with self.assertRaisesRegex(ValueError, "objective"):
            DistributedMinerWrapper(PairMarginMiner(), objective=objective)
