import unittest
from io import BytesIO

import torch

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import ContrastiveLoss, CrossBatchMemory
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

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
        for indices in (([0], [1], [2]), ([0], [1], [0], [2])):
            values = tuple(torch.tensor(x, device=TEST_DEVICE) for x in indices)
            actual = lmu.remove_self_comparisons(values, empty, 3)
            for observed, wanted in zip(actual, values, strict=True):
                torch.testing.assert_close(observed, wanted)
