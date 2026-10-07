import math
import unittest

import torch

from pytorch_metric_learning.losses import DynamicSoftMarginLoss

from .. import TEST_DEVICE, TEST_DTYPES


def embeddings_at_angles(angles, dtype):
    return torch.tensor(
        [(math.cos(angle), math.sin(angle)) for angle in angles],
        dtype=dtype,
        device=TEST_DEVICE,
        requires_grad=True,
    )


def scalar_histogram(margins, bins, min_val):
    """Independent Python probability-mass accumulation, outside autograd."""
    histogram = [0.0] * bins
    width = 2 * abs(min_val) / bins
    for margin in margins:
        position = (margin - min_val) / width
        low = math.floor(position)
        fraction = position - low
        histogram[low] += 1 - fraction
        histogram[low + 1] += fraction
    total = sum(histogram)
    return [value / total for value in histogram]


def independent_margins(embeddings):
    normalized = torch.nn.functional.normalize(embeddings, dim=1)
    positive = torch.norm(normalized[0] - normalized[1], p=2)
    return torch.stack(
        [
            positive - torch.norm(normalized[anchor] - normalized[2], p=2)
            for anchor in (0, 1)
        ]
    )


class TestDynamicSoftMarginHistogramAutograd(unittest.TestCase):
    def floating_dtypes(self):
        return [dtype for dtype in TEST_DTYPES if dtype != torch.float16]

    def test_repeated_public_loss_backward_does_not_reuse_freed_graph(self):
        for dtype in self.floating_dtypes():
            with self.subTest(dtype=dtype):
                loss_func = DynamicSoftMarginLoss(num_bins=16, momentum=0.25)
                labels = torch.tensor([0, 0, 1], device=TEST_DEVICE)
                for step in range(3):
                    embeddings = embeddings_at_angles(
                        (0.0, 0.7 + step * 0.05, 1.9), dtype
                    )
                    loss = loss_func(embeddings, labels)
                    loss.backward()
                    self.assertTrue(torch.isfinite(embeddings.grad).all())
                self.assertFalse(loss_func.hist_.requires_grad)
                self.assertIsNone(loss_func.hist_.grad_fn)

    def test_current_batch_gradient_uses_detached_cdf_weights(self):
        for dtype in self.floating_dtypes():
            with self.subTest(dtype=dtype):
                actual_embeddings = embeddings_at_angles((0.0, 0.7, 1.9), dtype)
                reference_embeddings = (
                    actual_embeddings.detach().clone().requires_grad_()
                )
                loss_func = DynamicSoftMarginLoss(num_bins=16, momentum=0.25)
                labels = torch.tensor([0, 0, 1], device=TEST_DEVICE)
                actual = loss_func(actual_embeddings, labels)
                margins = independent_margins(reference_embeddings)
                histogram = scalar_histogram(margins.detach().cpu().tolist(), 16, -2.0)
                weights = []
                for margin in margins.detach().cpu().tolist():
                    index = math.floor((margin + 2.0) / 0.25)
                    weights.append(sum(histogram[: index + 1]))
                expected = (margins * margins.new_tensor(weights)).mean()
                actual.backward()
                expected.backward()
                self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=1e-6))
                self.assertTrue(
                    torch.allclose(
                        actual_embeddings.grad,
                        reference_embeddings.grad,
                        rtol=1e-5,
                        atol=1e-6,
                    )
                )

    def test_history_keeps_probability_values_without_graph_nodes(self):
        for dtype in self.floating_dtypes():
            for momentum in (0.0, 0.25, 1.0):
                with self.subTest(dtype=dtype, momentum=momentum):
                    loss_func = DynamicSoftMarginLoss(num_bins=16, momentum=momentum)
                    labels = torch.tensor([0, 0, 1], device=TEST_DEVICE)
                    first_histogram = None
                    for step in range(3):
                        embeddings = embeddings_at_angles(
                            (0.0, 0.7 + step * 0.05, 1.9), dtype
                        )
                        loss_func(embeddings, labels)
                        margins = (
                            independent_margins(embeddings).detach().cpu().tolist()
                        )
                        current_histogram = scalar_histogram(margins, 16, -2.0)
                        if first_histogram is None:
                            first_histogram = current_histogram
                        if step == 0 or momentum in (0.0, 1.0):
                            expected = (
                                first_histogram
                                if momentum == 0.0
                                else current_histogram
                            )
                            self.assertTrue(
                                torch.allclose(
                                    loss_func.hist_,
                                    embeddings.new_tensor(expected),
                                    rtol=1e-5,
                                    atol=1e-6,
                                )
                            )
                        self.assertTrue(torch.isfinite(loss_func.hist_).all())
                        self.assertTrue((loss_func.hist_ >= 0).all())
                        self.assertTrue(
                            torch.allclose(
                                loss_func.hist_.sum(),
                                embeddings.new_tensor(1.0),
                                rtol=1e-5,
                                atol=1e-6,
                            )
                        )
                        self.assertFalse(loss_func.hist_.requires_grad)
                        self.assertIsNone(loss_func.hist_.grad_fn)

    def test_current_batch_backward_does_not_update_previous_embeddings(self):
        for dtype in self.floating_dtypes():
            with self.subTest(dtype=dtype):
                loss_func = DynamicSoftMarginLoss(num_bins=16, momentum=0.25)
                labels = torch.tensor([0, 0, 1], device=TEST_DEVICE)
                previous_embeddings = embeddings_at_angles((0.0, 0.7, 1.9), dtype)
                loss_func(previous_embeddings, labels)
                current_embeddings = embeddings_at_angles((0.0, 0.8, 1.9), dtype)
                loss_func(current_embeddings, labels).backward()
                self.assertIsNone(previous_embeddings.grad)
                self.assertIsNotNone(current_embeddings.grad)
                self.assertTrue(torch.isfinite(current_embeddings.grad).all())
