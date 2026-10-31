import unittest
from typing import Any

import torch
from torch.nn.functional import normalize

from pytorch_metric_learning.losses import DynamicSoftMarginLoss

from .. import TEST_DEVICE, TEST_DTYPES


def inputs(dtype: torch.dtype, count: int = 4) -> torch.Tensor:
    return torch.tensor(
        [[1.0, 0.1], [0.8, 0.4], [-0.5, 0.8], [-0.2, -1.0], [0.2, 0.7], [0.3, -0.9]],
        device=TEST_DEVICE,
        dtype=dtype,
    )[:count].requires_grad_()


def labeled_differences(embeddings: torch.Tensor, labels: list[int]) -> torch.Tensor:
    vectors = normalize(embeddings, dim=1)
    return torch.stack(
        [
            torch.linalg.vector_norm(vectors[a] - vectors[p])
            - torch.linalg.vector_norm(vectors[a] - vectors[n])
            for a in range(len(labels))
            for p in range(len(labels))
            for n in range(len(labels))
            if a != p and labels[a] == labels[p] and labels[a] != labels[n]
        ]
    )


def paired_differences(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    query, reference = normalize(query, dim=1), normalize(reference, dim=1)
    differences = []
    for i in range(min(len(query), len(reference))):
        positive = torch.linalg.vector_norm(query[i] - reference[i])
        candidates = [
            torch.linalg.vector_norm(query[i] - reference[j])
            for j in range(len(reference))
            if j != i
        ] + [
            torch.linalg.vector_norm(query[j] - reference[i])
            for j in range(len(query))
            if j != i
        ]
        differences.append(positive - torch.stack(candidates).min())
    return torch.stack(differences)


def histogram_reference(
    differences: torch.Tensor,
    previous: torch.Tensor | None = None,
    momentum: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = [-2.0, -1.0, 0.0, 1.0, 2.0]
    values = [max(-2.0, min(2.0, x)) for x in differences.detach().cpu().tolist()]
    probabilities = differences.new_tensor(
        [
            sum(max(0.0, 1 - abs(x - center)) for x in values) / len(values)
            for center in centers
        ]
    )
    if previous is not None:
        probabilities = (1 - momentum) * previous + momentum * probabilities
    weights = torch.stack(
        [probabilities[: sum(center <= x for center in centers)].sum() for x in values]
    )
    return (differences * weights).mean(), probabilities


def assert_numeric_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.float16:
        torch.testing.assert_close(
            actual.double(), expected.double(), atol=5e-4, rtol=5e-3
        )
    else:
        torch.testing.assert_close(actual, expected.to(actual.dtype))


class TestDynamicSoftMarginLoss(unittest.TestCase):
    def test_autocast_histogram_uses_buffer_precision(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                loss = DynamicSoftMarginLoss(num_bins=5, momentum=0.25).to(TEST_DEVICE)
                previous = None
                for shift in (0.0, 0.25):
                    data = (
                        torch.tensor(
                            [-1.5, -0.25, -0.0001, 0.5, 2.0],
                            device=TEST_DEVICE,
                            dtype=dtype,
                        )
                        .add(shift)
                        .requires_grad_()
                    )
                    expected, previous = histogram_reference(data.float(), previous)
                    with torch.autocast(TEST_DEVICE.type, dtype=dtype):
                        loss.update_histogram(data)
                        actual = loss.weigh_loss(data).mean()
                    self.assertEqual(loss.hist.dtype, torch.float32)
                    torch.testing.assert_close(loss.hist, previous)
                    torch.testing.assert_close(actual, expected)
                    torch.testing.assert_close(
                        torch.autograd.grad(actual, data)[0],
                        torch.autograd.grad(expected, data)[0],
                    )

    def test_autocast_training_steps_have_finite_gradients(self) -> None:
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(17)
                encoder = torch.nn.Linear(2, 2).to(TEST_DEVICE)
                optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01)
                loss = DynamicSoftMarginLoss().to(TEST_DEVICE)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(TEST_DEVICE.type, dtype=dtype):
                        embeddings = encoder(inputs(torch.float32))
                        self.assertEqual(embeddings.dtype, dtype)
                        value = loss(embeddings, labels)
                    value.backward()
                    self.assertTrue(torch.isfinite(value))
                    self.assertEqual(loss.hist.dtype, torch.float32)
                    for parameter in encoder.parameters():
                        gradient = parameter.grad
                        assert gradient is not None
                        self.assertTrue(torch.isfinite(gradient).all())
                        self.assertGreater(gradient.abs().sum().item(), 0)
                    optimizer.step()

    def test_values_and_gradients_across_training_steps(self) -> None:
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                loss = DynamicSoftMarginLoss(num_bins=5, momentum=0.25).to(
                    TEST_DEVICE, dtype
                )
                previous = None
                for count in (4, 6, 4):
                    embedding = inputs(dtype, count)
                    exact = embedding.detach().double().requires_grad_()
                    labels = [i // 2 for i in range(count)]
                    expected, previous = histogram_reference(
                        labeled_differences(exact, labels), previous
                    )
                    expected_gradient = torch.autograd.grad(expected, exact)[0]
                    actual = loss(embedding, torch.tensor(labels, device=TEST_DEVICE))
                    gradient = torch.autograd.grad(actual, embedding)[0]
                    assert_numeric_close(actual, expected)
                    assert_numeric_close(gradient, expected_gradient)
                    assert_numeric_close(loss.hist, previous)

    def test_paired_batches_and_accumulated_backward(self) -> None:
        for dtype in TEST_DTYPES:
            for query_count, reference_count in ((3, 3), (2, 3), (3, 2)):
                with self.subTest(dtype=dtype, sizes=(query_count, reference_count)):
                    loss = DynamicSoftMarginLoss(num_bins=5, momentum=0.25).to(
                        TEST_DEVICE, dtype
                    )
                    previous = None
                    actual_losses, expected_losses, variables = [], [], []
                    for shift in (0.05, 0.2):
                        query = inputs(dtype, query_count)
                        reference = (
                            inputs(dtype, reference_count).detach() + shift
                        ).requires_grad_()
                        expected, previous = histogram_reference(
                            paired_differences(query, reference), previous
                        )
                        expected_losses.append(expected)
                        actual_losses.append(loss(query, ref_emb=reference))
                        variables.extend((query, reference))
                    expected = torch.stack(expected_losses).sum()
                    actual = torch.stack(actual_losses).sum()
                    expected_gradients = torch.autograd.grad(expected, variables)
                    gradients = torch.autograd.grad(actual, variables)
                    assert_numeric_close(actual, expected)
                    for observed, wanted in zip(
                        gradients, expected_gradients, strict=True
                    ):
                        assert_numeric_close(observed, wanted)

    def test_reported_convolutional_training_loop(self) -> None:
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 32),
        ).to(TEST_DEVICE)
        self.assertEqual(model[0].weight.shape, (128, 128, 1, 1))
        loss_func = DynamicSoftMarginLoss().to(TEST_DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        labels = (torch.arange(16) // 4).to(TEST_DEVICE)
        for _ in range(4):
            data = torch.randn(16, 128, 4, 4, device=TEST_DEVICE)
            optimizer.zero_grad()
            loss_func(model(data), labels).backward()
            optimizer.step()
            self.assertFalse(loss_func.hist.requires_grad)
            self.assertIsNone(loss_func.hist.grad_fn)

    def test_earlier_graph_stays_out_of_later_steps(self) -> None:
        loss_func = DynamicSoftMarginLoss(num_bins=5).to(TEST_DEVICE)
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        first = inputs(torch.float32)
        loss_func(first, labels).backward()
        first_gradient = first.grad
        assert first_gradient is not None
        expected = first_gradient.clone()
        second = inputs(torch.float32)
        loss_func(second, labels).backward()
        self.assertTrue(torch.equal(first_gradient, expected))
        self.assertIsNotNone(second.grad)

    def test_forwards_before_backward_in_any_order(self) -> None:
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        for order in ((0, 1), (1, 0)):
            with self.subTest(order=order):
                loss_func = DynamicSoftMarginLoss(num_bins=5).to(TEST_DEVICE)
                rows = [inputs(torch.float32), inputs(torch.float32) * 2]
                values = [loss_func(row, labels) for row in rows]
                for index in order:
                    values[index].backward()

    def test_forwards_summed_before_backward_match_separate_backward(self) -> None:
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        results = []
        for summed in (True, False):
            loss_func = DynamicSoftMarginLoss(num_bins=5).to(TEST_DEVICE)
            embeddings = inputs(torch.float64)
            loss_func = loss_func.to(TEST_DEVICE, torch.float64)
            values = [loss_func(embeddings * s, labels) for s in (1.0, 0.5)]
            if summed:
                (values[0] + values[1]).backward()
            else:
                values[0].backward(retain_graph=True)
                values[1].backward()
            results.append(embeddings.grad)
        assert results[0] is not None and results[1] is not None
        torch.testing.assert_close(results[0], results[1])

    def test_histogram_interpolation_endpoints_and_clipping(self) -> None:
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                data = torch.tensor(
                    [-3.0, -1.5, -0.25, 0.5, 2.0, 3.0],
                    device=TEST_DEVICE,
                    dtype=dtype,
                    requires_grad=True,
                )
                loss = DynamicSoftMarginLoss(num_bins=5).to(TEST_DEVICE, dtype)
                loss.update_histogram(data)
                actual = loss.weigh_loss(data).mean()
                expected, histogram = histogram_reference(data)
                torch.testing.assert_close(loss.hist, histogram)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(
                    torch.autograd.grad(actual, data)[0],
                    torch.autograd.grad(expected, data)[0],
                )

    def test_supplied_triplets_without_labels_keep_reference_gradients(self) -> None:
        query = inputs(torch.float64)
        reference = (inputs(torch.float64, 3).detach() - 0.1).requires_grad_()
        q, r = normalize(query, dim=1), normalize(reference, dim=1)
        differences = torch.stack(
            [
                torch.linalg.vector_norm(q[0] - r[1])
                - torch.linalg.vector_norm(q[0] - r[0]),
                torch.linalg.vector_norm(q[1] - r[2])
                - torch.linalg.vector_norm(q[1] - r[0]),
            ]
        )
        expected, _ = histogram_reference(differences)
        indices = tuple(
            torch.tensor(i, device=TEST_DEVICE) for i in ([0, 1], [1, 2], [0, 0])
        )
        actual = DynamicSoftMarginLoss(num_bins=5).to(TEST_DEVICE, torch.float64)(
            query, indices_tuple=indices, ref_emb=reference
        )
        expected_gradients = torch.autograd.grad(expected, (query, reference))
        gradients = torch.autograd.grad(actual, (query, reference))
        torch.testing.assert_close(actual, expected)
        for observed, wanted in zip(gradients, expected_gradients, strict=True):
            torch.testing.assert_close(observed, wanted)

    def test_coincident_and_opposite_embeddings_have_finite_gradients(self) -> None:
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                embeddings = torch.tensor(
                    [[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
                    device=TEST_DEVICE,
                    dtype=dtype,
                    requires_grad=True,
                )
                labels = [0, 0, 1, 1]
                expected, _ = histogram_reference(
                    labeled_differences(embeddings, labels)
                )
                actual = DynamicSoftMarginLoss(num_bins=5).to(TEST_DEVICE, dtype)(
                    embeddings, torch.tensor(labels, device=TEST_DEVICE)
                )
                expected_gradient = torch.autograd.grad(expected, embeddings)[0]
                gradient = torch.autograd.grad(actual, embeddings)[0]
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(gradient, expected_gradient)

    def test_empty_valid_sets_preserve_histogram(self) -> None:
        loss = DynamicSoftMarginLoss(num_bins=5).to(TEST_DEVICE)
        loss(inputs(torch.float32), torch.tensor([0, 0, 1, 1], device=TEST_DEVICE))
        previous = loss.hist.detach().clone()
        for query_count, reference_count in ((0, 2), (2, 0), (1, 1)):
            query = inputs(torch.float32, query_count)
            actual = loss(query, ref_emb=inputs(torch.float32, reference_count))
            self.assertEqual(actual.item(), 0)
            torch.testing.assert_close(
                torch.autograd.grad(actual, query)[0], torch.zeros_like(query)
            )
            torch.testing.assert_close(loss.hist, previous)
        for labels in ([0, 0, 0, 0], [0, 1, 2, 3]):
            query = inputs(torch.float32)
            actual = loss(query, torch.tensor(labels, device=TEST_DEVICE))
            self.assertEqual(actual.item(), 0)
            torch.testing.assert_close(
                torch.autograd.grad(actual, query)[0], torch.zeros_like(query)
            )
            torch.testing.assert_close(loss.hist, previous)

    def test_checkpoint_preserves_next_loss_and_gradient(self) -> None:
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        original = DynamicSoftMarginLoss(num_bins=5, momentum=0.25).to(
            TEST_DEVICE, torch.float64
        )
        original(inputs(torch.float64), labels).backward()
        restored = DynamicSoftMarginLoss(num_bins=5, momentum=0.25).to(
            TEST_DEVICE, torch.float64
        )
        restored.load_state_dict(original.state_dict(), assign=True)
        original_input, restored_input = inputs(torch.float64), inputs(torch.float64)
        expected, actual = (
            original(original_input, labels),
            restored(restored_input, labels),
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, restored_input)[0],
            torch.autograd.grad(expected, original_input)[0],
        )

    def test_invalid_histogram_configuration(self) -> None:
        invalid: tuple[dict[str, Any], ...] = (
            {"min_val": 0},
            {"min_val": float("nan")},
            {"min_val": -float("inf")},
            {"num_bins": 1},
            {"momentum": -0.1},
            {"momentum": 1.1},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DynamicSoftMarginLoss(**kwargs).to(TEST_DEVICE)

    def test_half_precision_histogram_counts_more_triplets_than_float16_max(
        self,
    ) -> None:
        generator = torch.Generator().manual_seed(0)
        embeddings = torch.randn(128, 32, generator=generator).to(TEST_DEVICE)
        labels = torch.arange(128, device=TEST_DEVICE) % 2
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                expected = DynamicSoftMarginLoss().to(TEST_DEVICE, torch.float64)
                actual = DynamicSoftMarginLoss().to(TEST_DEVICE, dtype)
                for _ in range(2):
                    reference = expected(embeddings.double(), labels)
                    loss = actual(embeddings.to(dtype), labels)
                self.assertEqual(loss.dtype, dtype)
                torch.testing.assert_close(
                    loss.double(), reference, rtol=2e-2, atol=1e-3
                )
                torch.testing.assert_close(
                    actual.hist.double(), expected.hist, rtol=2e-2, atol=1e-3
                )
