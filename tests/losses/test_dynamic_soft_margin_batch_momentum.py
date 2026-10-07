import math
import unittest

import torch

from pytorch_metric_learning.losses import DynamicSoftMarginLoss

from .. import TEST_DEVICE, TEST_DTYPES


def probability_mass(values, bins=16, minimum=-2.0):
    """Build a normalized batch histogram with independent scalar arithmetic."""
    result = [0.0] * bins
    width = 2 * abs(minimum) / bins
    for value in values:
        position = (value - minimum) / width
        lower = math.floor(position)
        fraction = position - lower
        result[lower] += 1 - fraction
        result[lower + 1] += fraction
    return [value / len(values) for value in result]


class TestDynamicSoftMarginBatchMomentum(unittest.TestCase):
    def floating_dtypes(self):
        return [dtype for dtype in TEST_DTYPES if dtype != torch.float16]

    def assert_histogram_close(self, actual, expected):
        target = actual.new_tensor(expected)
        tolerance = 2e-7 if actual.dtype == torch.float32 else 1e-12
        self.assertTrue(torch.allclose(actual, target, atol=tolerance, rtol=tolerance))

    def test_batch_histogram_has_the_declared_mixture_weight(self):
        old_values = [-1.125, -0.625]
        new_values = [0.125, 0.625]
        old_pdf = probability_mass(old_values)
        new_pdf = probability_mass(new_values)
        for dtype in self.floating_dtypes():
            for momentum in (0.1, 0.25, 0.75):
                with self.subTest(dtype=dtype, momentum=momentum):
                    loss = DynamicSoftMarginLoss(num_bins=16, momentum=momentum)
                    loss.hist_ = torch.zeros(16, dtype=dtype, device=TEST_DEVICE)
                    loss.update_histogram(
                        torch.tensor(old_values, dtype=dtype, device=TEST_DEVICE)
                    )
                    loss.update_histogram(
                        torch.tensor(new_values, dtype=dtype, device=TEST_DEVICE)
                    )
                    expected = [
                        (1 - momentum) * old + momentum * new
                        for old, new in zip(old_pdf, new_pdf)
                    ]
                    self.assert_histogram_close(loss.hist_, expected)

    def test_repeating_current_batch_preserves_momentum(self):
        for dtype in self.floating_dtypes():
            histograms = []
            for repetitions in (1, 3, 16):
                loss = DynamicSoftMarginLoss(num_bins=16, momentum=0.25)
                loss.hist_ = torch.zeros(16, dtype=dtype, device=TEST_DEVICE)
                loss.update_histogram(
                    torch.tensor([-1.125, -0.625], dtype=dtype, device=TEST_DEVICE)
                )
                loss.update_histogram(
                    torch.tensor(
                        [0.125, 0.625] * repetitions,
                        dtype=dtype,
                        device=TEST_DEVICE,
                    )
                )
                histograms.append(loss.hist_.detach().clone())
            with self.subTest(dtype=dtype):
                for histogram in histograms[1:]:
                    self.assert_histogram_close(histogram, histograms[0].tolist())

    def test_variable_batch_history_matches_probability_mixture(self):
        batches = [
            [-1.125, -0.625],
            [0.125, 0.625, 0.125, 0.625, 0.125, 0.625],
            [-0.375, 0.375, 0.875],
        ]
        momentum = 0.3
        for dtype in self.floating_dtypes():
            with self.subTest(dtype=dtype):
                loss = DynamicSoftMarginLoss(num_bins=16, momentum=momentum)
                loss.hist_ = torch.zeros(16, dtype=dtype, device=TEST_DEVICE)
                expected = None
                for values in batches:
                    current = probability_mass(values)
                    expected = (
                        current
                        if expected is None
                        else [
                            (1 - momentum) * old + momentum * new
                            for old, new in zip(expected, current)
                        ]
                    )
                    loss.update_histogram(
                        torch.tensor(values, dtype=dtype, device=TEST_DEVICE)
                    )
                    self.assert_histogram_close(loss.hist_, expected)
                    self.assertAlmostEqual(loss.hist_.sum().item(), 1.0, places=6)

    def test_zero_and_unit_momentum_controls(self):
        old_values = [-1.125, -0.625]
        new_values = [0.125, 0.625]
        for dtype in self.floating_dtypes():
            for momentum in (0.0, 1.0):
                for repetitions in (1, 7):
                    with self.subTest(
                        dtype=dtype,
                        momentum=momentum,
                        repetitions=repetitions,
                    ):
                        loss = DynamicSoftMarginLoss(num_bins=16, momentum=momentum)
                        loss.hist_ = torch.zeros(16, dtype=dtype, device=TEST_DEVICE)
                        loss.update_histogram(
                            torch.tensor(old_values, dtype=dtype, device=TEST_DEVICE)
                        )
                        self.assert_histogram_close(
                            loss.hist_, probability_mass(old_values)
                        )
                        loss.update_histogram(
                            torch.tensor(
                                new_values * repetitions,
                                dtype=dtype,
                                device=TEST_DEVICE,
                            )
                        )
                        self.assert_histogram_close(
                            loss.hist_,
                            probability_mass(
                                old_values if momentum == 0.0 else new_values
                            ),
                        )
