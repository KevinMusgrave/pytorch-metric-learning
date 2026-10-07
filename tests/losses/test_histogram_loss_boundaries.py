import unittest

import torch

from pytorch_metric_learning.losses import HistogramLoss

from .. import TEST_DEVICE, TEST_DTYPES


class TestHistogramLossBoundaries(unittest.TestCase):
    def assert_probability_close(self, actual, expected):
        rtol = 1e-2 if actual.dtype == torch.float16 else 1e-5
        self.assertTrue(torch.allclose(actual, expected, rtol=rtol, atol=1e-6))

    def test_endpoint_probability_mass_is_conserved(self):
        for dtype in TEST_DTYPES:
            for bins in (1, 4, 20, 100):
                with self.subTest(dtype=dtype, bins=bins):
                    loss = HistogramLoss(n_bins=bins)
                    distances = torch.tensor(
                        [-1.0, 1.0], dtype=dtype, device=TEST_DEVICE, requires_grad=True
                    )
                    actual = loss.compute_density(distances)
                    expected = torch.zeros(bins + 1, dtype=dtype, device=TEST_DEVICE)
                    expected[0], expected[-1] = 0.5, 0.5
                    self.assert_probability_close(actual, expected)
                    self.assert_probability_close(
                        actual.sum(), distances.new_tensor(1.0)
                    )
                    (
                        actual * torch.arange(bins + 1, device=TEST_DEVICE, dtype=dtype)
                    ).sum().backward()
                    self.assertTrue(torch.isfinite(distances.grad).all())

    def test_collapsed_embeddings_have_unit_loss_without_index_errors(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                embeddings = torch.tensor(
                    [[1.0, 0.0]] * 4,
                    dtype=dtype,
                    device=TEST_DEVICE,
                    requires_grad=True,
                )
                labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
                actual = HistogramLoss()(embeddings, labels)
                self.assert_probability_close(actual, embeddings.new_tensor(1.0))
                actual.backward()
                self.assertTrue(torch.isfinite(embeddings.grad).all())

    def test_opposite_clusters_have_zero_ranking_probability(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                embeddings = torch.tensor(
                    [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
                    dtype=dtype,
                    device=TEST_DEVICE,
                    requires_grad=True,
                )
                labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
                actual = HistogramLoss(n_bins=4)(embeddings, labels)
                self.assert_probability_close(actual, embeddings.new_tensor(0.0))
                actual.backward()
                self.assertTrue(torch.isfinite(embeddings.grad).all())

    def test_interior_interpolation_preserves_values_and_slopes(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                distances = torch.tensor(
                    [-0.75, 0.25], dtype=dtype, device=TEST_DEVICE, requires_grad=True
                )
                actual = HistogramLoss(n_bins=4).compute_density(distances)
                expected = distances.new_tensor([0.25, 0.25, 0.25, 0.25, 0.0])
                self.assert_probability_close(actual, expected)
                projection = (
                    actual * distances.new_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
                ).sum()
                projection.backward()
                self.assert_probability_close(
                    distances.grad, torch.ones_like(distances)
                )

    def test_delta_parameter_uses_the_same_closed_support(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                distances = torch.tensor(
                    [-1.0, 0.0, 1.0], dtype=dtype, device=TEST_DEVICE
                )
                actual = HistogramLoss(delta=0.5).compute_density(distances)
                expected = distances.new_tensor([1 / 3, 0.0, 1 / 3, 0.0, 1 / 3])
                self.assert_probability_close(actual, expected)
