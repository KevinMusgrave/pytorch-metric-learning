import unittest
from io import BytesIO
from typing import Any

import torch

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import (
    ContrastiveLoss,
    CrossBatchMemory,
    LiftedStructureLoss,
)
from pytorch_metric_learning.types import PairIndices, TripletIndices
from pytorch_metric_learning.utils import indexing

from .. import TEST_DEVICE, TEST_DTYPES


def make_loss() -> ContrastiveLoss:
    return ContrastiveLoss(distance=LpDistance(normalize_embeddings=False))


def make_inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = torch.tensor(
        [[0.0, 0.0], [0.2, 0.1], [0.3, -0.1], [0.8, 0.2]],
        dtype=dtype,
        device=TEST_DEVICE,
        requires_grad=True,
    )
    return embeddings, torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)


def reference_pairs(
    labels: torch.Tensor, reference_labels: torch.Tensor, offset: int
) -> tuple[torch.Tensor, ...]:
    positive = labels[:, None] == reference_labels[None, :]
    negative = ~positive
    indices = torch.arange(len(labels), device=labels.device)
    positive[indices, indices + offset] = False
    return (*torch.where(positive), *torch.where(negative))


class TestCrossBatchMemoryState(unittest.TestCase):
    def test_autocast_preserves_queue_dtype_and_anchor_gradients(self) -> None:
        inputs, labels = make_inputs(torch.float32)
        mask = torch.tensor([True, True, False, False], device=TEST_DEVICE)
        for dtype in (torch.float16, torch.bfloat16):
            for enqueue_mask in (None, mask):
                with self.subTest(dtype=dtype, masked=enqueue_mask is not None):
                    encoder = torch.nn.Linear(2, 2, bias=False).to(TEST_DEVICE)
                    with torch.no_grad():
                        encoder.weight.copy_(
                            inputs.new_tensor([[0.75, 0.25], [-0.25, 0.5]])
                        )
                    loss = CrossBatchMemory(
                        make_loss(), embedding_size=2, memory_size=8
                    ).to(TEST_DEVICE)
                    stored, stored_labels = [], []
                    for step in range(2):
                        with torch.autocast(TEST_DEVICE.type, dtype=dtype):
                            embeddings = encoder(inputs.detach() + step * 0.1)
                            self.assertEqual(embeddings.dtype, dtype)
                            actual = loss(embeddings, labels, enqueue_mask=enqueue_mask)
                        current = embeddings.float()
                        enqueued = current if enqueue_mask is None else current[mask]
                        queued_labels = labels if enqueue_mask is None else labels[mask]
                        stored.append(enqueued.detach())
                        stored_labels.append(queued_labels)
                        references = torch.cat(stored)
                        reference_labels = torch.cat(stored_labels)
                        pairs = (
                            reference_pairs(
                                labels, reference_labels, step * len(labels)
                            )
                            if enqueue_mask is None
                            else None
                        )
                        expected = make_loss()(
                            current if enqueue_mask is None else current[~mask],
                            labels if enqueue_mask is None else labels[~mask],
                            pairs,
                            references,
                            reference_labels,
                        )
                        expected_gradient = torch.autograd.grad(expected, embeddings)[0]
                        gradient = torch.autograd.grad(actual, embeddings)[0]
                        torch.testing.assert_close(actual, expected)
                        torch.testing.assert_close(gradient, expected_gradient)
                        self.assertGreater(gradient.abs().sum().item(), 0)
                        self.assertEqual(loss.embedding_memory.dtype, torch.float32)
                        self.assertFalse(loss.embedding_memory.requires_grad)
                        torch.testing.assert_close(
                            loss.embedding_memory[: len(references)], references
                        )

    def assert_queue_unchanged(
        self,
        loss: CrossBatchMemory,
        memory: torch.Tensor,
        labels: torch.Tensor,
        cursor: int,
        filled: bool,
    ) -> None:
        torch.testing.assert_close(loss.embedding_memory, memory)
        torch.testing.assert_close(loss.label_memory, labels)
        self.assertEqual(loss.queue_idx, cursor)
        self.assertEqual(loss.has_been_filled, filled)

    def test_empty_enqueue_uses_only_existing_memory(self) -> None:
        for dtype in TEST_DTYPES:
            for preload in (False, True):
                with self.subTest(dtype=dtype, preload=preload):
                    inputs, labels = make_inputs(dtype)
                    loss = CrossBatchMemory(
                        make_loss(), embedding_size=2, memory_size=8
                    ).to(TEST_DEVICE, dtype)
                    if preload:
                        loss(inputs.detach(), labels)
                    memory, memory_labels = (
                        loss.embedding_memory.clone(),
                        loss.label_memory.clone(),
                    )
                    cursor, filled = loss.queue_idx, loss.has_been_filled
                    expected = make_loss()(
                        inputs,
                        labels,
                        ref_emb=memory[:cursor],
                        ref_labels=memory_labels[:cursor],
                    )
                    expected_gradient = torch.autograd.grad(expected, inputs)[0]
                    actual = loss(
                        inputs,
                        labels,
                        enqueue_mask=torch.zeros(
                            4, device=TEST_DEVICE, dtype=torch.bool
                        ),
                    )
                    actual_gradient = torch.autograd.grad(actual, inputs)[0]
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(actual_gradient, expected_gradient)
                    self.assert_queue_unchanged(
                        loss, memory, memory_labels, cursor, filled
                    )

    def test_empty_batch_preserves_queue_and_has_zero_gradient(self) -> None:
        for dtype in TEST_DTYPES:
            for preload in (False, True):
                with self.subTest(dtype=dtype, preload=preload):
                    inputs, labels = make_inputs(dtype)
                    loss = CrossBatchMemory(
                        make_loss(), embedding_size=2, memory_size=4
                    ).to(TEST_DEVICE, dtype)
                    if preload:
                        loss(inputs.detach(), labels)
                    memory, memory_labels = (
                        loss.embedding_memory.clone(),
                        loss.label_memory.clone(),
                    )
                    cursor, filled = loss.queue_idx, loss.has_been_filled
                    empty = inputs.detach()[:0].requires_grad_()
                    actual = loss(empty, labels[:0])
                    gradient = torch.autograd.grad(actual, empty)[0]
                    self.assertEqual(actual.item(), 0)
                    torch.testing.assert_close(gradient, torch.zeros_like(empty))
                    self.assert_queue_unchanged(
                        loss, memory, memory_labels, cursor, filled
                    )

    def test_multiple_forwards_keep_backward_references_stable(self) -> None:
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                first, labels = make_inputs(dtype)
                second = (first.detach() + 0.15).requires_grad_()
                first_memory = first.detach().clone()
                second_memory = torch.cat((first.detach(), second.detach()))
                second_labels = labels.repeat(2)
                expected = make_loss()(
                    first,
                    labels,
                    reference_pairs(labels, labels, 0),
                    first_memory,
                    labels,
                )
                expected = expected + make_loss()(
                    second,
                    labels,
                    reference_pairs(labels, second_labels, 4),
                    second_memory,
                    second_labels,
                )
                expected_gradients = torch.autograd.grad(expected, (first, second))
                memory = CrossBatchMemory(
                    make_loss(), embedding_size=2, memory_size=8
                ).to(TEST_DEVICE, dtype)
                actual = memory(first, labels) + memory(second, labels)
                gradients = torch.autograd.grad(actual, (first, second))
                self.assertGreater(actual.item(), 0)
                torch.testing.assert_close(actual, expected)
                for observed, wanted in zip(gradients, expected_gradients, strict=True):
                    torch.testing.assert_close(observed, wanted)
                    self.assertGreater(observed.abs().sum().item(), 0)

    def test_mask_cannot_enqueue_beyond_capacity(self) -> None:
        inputs, labels = make_inputs(torch.float32)
        loss = CrossBatchMemory(make_loss(), embedding_size=2, memory_size=2).to(
            TEST_DEVICE
        )
        memory, memory_labels = loss.embedding_memory.clone(), loss.label_memory.clone()
        with self.assertRaisesRegex(ValueError, "memory_size"):
            loss(
                inputs,
                labels,
                enqueue_mask=torch.ones(4, device=TEST_DEVICE, dtype=torch.bool),
            )
        self.assert_queue_unchanged(loss, memory, memory_labels, 0, False)

    def test_checkpoint_restores_the_next_loss_and_gradients(self) -> None:
        for dtype in TEST_DTYPES:
            for updates in (1, 2):
                with self.subTest(dtype=dtype, updates=updates):
                    inputs, labels = make_inputs(dtype)
                    labels = labels.to(torch.float64) + 0.25
                    original = CrossBatchMemory(
                        make_loss(),
                        embedding_size=2,
                        memory_size=6,
                        label_dtype=torch.float64,
                    ).to(TEST_DEVICE, dtype)
                    for step in range(updates):
                        original(inputs.detach() + step * 0.1, labels)
                    with BytesIO() as checkpoint:
                        torch.save(original.state_dict(), checkpoint)
                        checkpoint.seek(0)
                        state = torch.load(
                            checkpoint, weights_only=True, map_location=TEST_DEVICE
                        )
                    restored = CrossBatchMemory(
                        make_loss(), embedding_size=2, memory_size=6
                    ).to(TEST_DEVICE, dtype)
                    restored.load_state_dict(state, assign=True)
                    self.assert_queue_unchanged(
                        restored,
                        original.embedding_memory,
                        original.label_memory,
                        original.queue_idx,
                        original.has_been_filled,
                    )
                    uninterrupted = (inputs.detach() + 0.4).requires_grad_()
                    resumed = uninterrupted.detach().clone().requires_grad_()
                    expected = original(uninterrupted, labels)
                    actual = restored(resumed, labels)
                    expected_gradient = torch.autograd.grad(expected, uninterrupted)[0]
                    actual_gradient = torch.autograd.grad(actual, resumed)[0]
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(actual_gradient, expected_gradient)
                    self.assert_queue_unchanged(
                        restored,
                        original.embedding_memory,
                        original.label_memory,
                        original.queue_idx,
                        original.has_been_filled,
                    )

    def test_empty_self_indices_leave_pair_and_triplet_indices_unchanged(self) -> None:
        empty = torch.empty(0, dtype=torch.long, device=TEST_DEVICE)
        for kind, indices in (
            (TripletIndices, ([0], [1], [2])),
            (PairIndices, ([0], [1], [0], [2])),
        ):
            values: Any = kind(*(torch.tensor(x, device=TEST_DEVICE) for x in indices))
            actual = indexing.remove_self_comparisons(values, empty)
            for observed, wanted in zip(actual, values, strict=True):
                torch.testing.assert_close(observed, wanted)


class TestCrossBatchMemoryFailedCalls(unittest.TestCase):
    def assert_unchanged(self, memory: CrossBatchMemory, before: tuple) -> None:
        embeddings, labels, queue_idx, filled = before
        torch.testing.assert_close(memory.embedding_memory, embeddings)
        torch.testing.assert_close(memory.label_memory, labels)
        self.assertEqual(memory.queue_idx, queue_idx)
        self.assertEqual(memory.has_been_filled, filled)

    def snapshot(self, memory: CrossBatchMemory) -> tuple:
        return (
            memory.embedding_memory.clone(),
            memory.label_memory.clone(),
            memory.queue_idx,
            memory.has_been_filled,
        )

    def test_lossy_label_conversion_is_rejected_before_mutation(self) -> None:
        memory = CrossBatchMemory(
            ContrastiveLoss(), 2, memory_size=4, label_dtype=torch.int32
        ).to(TEST_DEVICE)
        before = self.snapshot(memory)
        labels = torch.tensor([0, 2**32], device=TEST_DEVICE)
        with self.assertRaisesRegex(ValueError, "label_dtype"):
            memory(torch.randn(2, 2, device=TEST_DEVICE), labels)
        self.assert_unchanged(memory, before)
        memory(
            torch.randn(2, 2, device=TEST_DEVICE),
            torch.tensor([0, 2**31 - 1], device=TEST_DEVICE),
        )
        self.assertEqual(memory.label_memory.dtype, torch.int32)

    def test_invalid_inputs_leave_queue_unchanged(self) -> None:
        memory = CrossBatchMemory(ContrastiveLoss(), 2, memory_size=4).to(TEST_DEVICE)
        memory(
            torch.randn(1, 2, device=TEST_DEVICE),
            torch.tensor([0], device=TEST_DEVICE),
        )
        before = self.snapshot(memory)
        with self.assertRaises(ValueError):
            memory(
                torch.randn(2, 2, device=TEST_DEVICE),
                torch.tensor([0], device=TEST_DEVICE),
            )
        self.assert_unchanged(memory, before)

    def test_wrapped_loss_errors_leave_queue_unchanged(self) -> None:
        memory = CrossBatchMemory(LiftedStructureLoss(), 2, memory_size=4)
        memory = memory.to(TEST_DEVICE)
        before = self.snapshot(memory)
        with self.assertRaisesRegex(ValueError, "ref_emb"):
            memory(
                torch.randn(2, 2, device=TEST_DEVICE),
                torch.tensor([0, 1], device=TEST_DEVICE),
            )
        self.assert_unchanged(memory, before)
