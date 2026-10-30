import unittest
from typing import Any

import torch
from numpy.testing import assert_almost_equal

from pytorch_metric_learning.losses import HistogramLoss

from .. import TEST_DEVICE, TEST_DTYPES, helpers


class OriginalImplementationHistogramLoss(torch.nn.Module):
    def __init__(self, num_steps):
        super().__init__()
        self.step = 2 / (num_steps - 1)
        self.eps = 1 / num_steps
        self.t = torch.arange(-1, 1 + self.step, self.step).view(-1, 1)
        self.tsize = self.t.size()[0]

    def forward(self, features, classes):
        def histogram(inds, size):
            s_repeat_ = s_repeat.clone()
            inds = inds.to(device=s_repeat_floor.device)
            self.t = self.t.to(device=s_repeat_floor.device)
            indsa = (
                (s_repeat_floor - (self.t - self.step) > -self.eps)
                & (s_repeat_floor - (self.t - self.step) < self.eps)
                & inds
            )
            assert indsa.nonzero().size()[0] == size, (
                "Another number of bins should be used"
            )
            zeros = torch.zeros((1, indsa.size()[1])).to(
                device=indsa.device, dtype=torch.uint8
            )
            indsb = torch.cat((indsa, zeros))[1:, :].to(dtype=torch.bool)
            s_repeat_[~(indsb | indsa)] = 0
            self.t = self.t.to(dtype=s_repeat_.dtype)
            s_repeat_[indsa] = (s_repeat_ - self.t + self.step)[indsa] / self.step
            s_repeat_[indsb] = (-s_repeat_ + self.t + self.step)[indsb] / self.step

            return s_repeat_.sum(1) / size

        classes_size = classes.size()[0]
        classes_eq = (
            classes.repeat(classes_size, 1)
            == classes.view(-1, 1).repeat(1, classes_size)
        ).detach()
        dists = torch.mm(features, features.transpose(0, 1))
        assert (
            (dists > 1 + self.eps).sum().item() + (dists < -1 - self.eps).sum().item()
        ) == 0, "L2 normalization should be used"
        s_inds = torch.triu(
            torch.ones(classes_eq.size(), dtype=torch.bool, device=features.device), 1
        )
        classes_eq = classes_eq.to(device=s_inds.device)
        pos_inds = classes_eq[s_inds].repeat(self.tsize, 1)
        neg_inds = ~classes_eq[s_inds].repeat(self.tsize, 1)
        pos_size = classes_eq[s_inds].sum().item()
        neg_size = (~classes_eq[s_inds]).sum().item()
        s = dists[s_inds].view(1, -1)
        s_repeat = s.repeat(self.tsize, 1)
        s_repeat_floor = (
            torch.floor(s_repeat.detach() / self.step) * self.step
        ).float()

        histogram_pos = histogram(pos_inds, pos_size)
        assert_almost_equal(
            histogram_pos.sum().item(),
            1,
            decimal=1,
            err_msg="Not good positive histogram",
            verbose=True,
        )
        histogram_neg = histogram(neg_inds, neg_size)
        assert_almost_equal(
            histogram_neg.sum().item(),
            1,
            decimal=1,
            err_msg="Not good negative histogram",
            verbose=True,
        )
        histogram_pos_repeat = histogram_pos.view(-1, 1).repeat(
            1, histogram_pos.size()[0]
        )
        histogram_pos_inds = torch.tril(
            torch.ones(histogram_pos_repeat.size()), -1
        ).bool()
        histogram_pos_inds = histogram_pos_inds.to(features.device)
        histogram_pos_repeat[histogram_pos_inds] = 0
        histogram_pos_cdf = histogram_pos_repeat.sum(0)
        loss = torch.sum(histogram_neg * histogram_pos_cdf)

        return loss


class TestHistogramLoss(unittest.TestCase):
    def test_histogram_loss(self):
        batch_size = 32
        embedding_size = 64
        for dtype in TEST_DTYPES:
            num_steps = 5 if dtype == torch.float16 else 21
            num_bins = num_steps - 1
            loss_func = HistogramLoss(num_bins=num_bins)
            original_loss_func = OriginalImplementationHistogramLoss(
                num_steps=num_steps
            )

            for _ in range(2):
                embeddings = torch.randn(
                    batch_size,
                    embedding_size,
                    requires_grad=True,
                    dtype=dtype,
                ).to(TEST_DEVICE)
                labels = torch.randint(0, 5, size=(batch_size,)).to(TEST_DEVICE)

                loss = loss_func(embeddings, labels)
                correct_loss = original_loss_func(
                    torch.nn.functional.normalize(embeddings), labels
                )

                rtol = helpers.rtol(dtype)
                self.assertTrue(torch.isclose(loss, correct_loss, rtol=rtol))

                loss.backward()

    def test_with_no_valid_triplets(self):
        loss_func = HistogramLoss(num_bins=4)
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(
                5,
                32,
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long).to(TEST_DEVICE)
            loss = loss_func(embeddings, labels)
            self.assertEqual(loss, 0)
            loss.backward()

    def test_assertion_raises(self):
        with self.assertRaises(ValueError):
            _ = HistogramLoss(num_bins=0)

        with self.assertRaises(TypeError):
            fractional: Any = 2.5
            _ = HistogramLoss(num_bins=fractional)


class TestHistogramEndpoints(unittest.TestCase):
    def test_identical_and_opposite_embeddings(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                embeddings = torch.tensor(
                    [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
                    dtype=dtype,
                    device=TEST_DEVICE,
                    requires_grad=True,
                )
                labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
                loss = HistogramLoss(num_bins=4)(embeddings, labels)
                (gradient,) = torch.autograd.grad(loss, embeddings)
                self.assertEqual(loss.item(), 0)
                self.assertTrue(torch.isfinite(gradient).all())
                loss = HistogramLoss(num_bins=4)(
                    embeddings, torch.tensor([0, 1, 0, 1], device=TEST_DEVICE)
                )
                self.assertEqual(loss.item(), 1)

    def test_collapsed_embeddings_have_unit_loss(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                embeddings = torch.tensor(
                    [[1.0, 0.0]] * 4,
                    dtype=dtype,
                    device=TEST_DEVICE,
                    requires_grad=True,
                )
                labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
                loss = HistogramLoss()(embeddings, labels)
                (gradient,) = torch.autograd.grad(loss, embeddings)
                helpers.assert_close(loss, 1.0, dtype)
                self.assertTrue(torch.isfinite(gradient).all())

    def test_endpoint_mass_splits_between_first_and_last_nodes(self):
        for dtype in TEST_DTYPES:
            for num_bins in (1, 4, 20, 100):
                with self.subTest(dtype=dtype, num_bins=num_bins):
                    distances = torch.tensor(
                        [-1.0, 1.0], dtype=dtype, device=TEST_DEVICE, requires_grad=True
                    )
                    density = HistogramLoss(num_bins=num_bins).compute_density(
                        distances
                    )
                    expected = torch.zeros(num_bins + 1, dtype=dtype)
                    expected[0], expected[-1] = 0.5, 0.5
                    helpers.assert_close(density, expected, dtype)
                    helpers.assert_close(density.sum(), 1.0, dtype)
                    nodes = torch.arange(num_bins + 1, dtype=dtype, device=TEST_DEVICE)
                    (gradient,) = torch.autograd.grad(density @ nodes, distances)
                    self.assertTrue(torch.isfinite(gradient).all())

    def test_interior_interpolation_preserves_values_and_slopes(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                distances = torch.tensor(
                    [-0.75, 0.25], dtype=dtype, device=TEST_DEVICE, requires_grad=True
                )
                density = HistogramLoss(num_bins=4).compute_density(distances)
                helpers.assert_close(density, [0.25, 0.25, 0.25, 0.25, 0.0], dtype)
                nodes = torch.arange(5, dtype=dtype, device=TEST_DEVICE)
                (gradient,) = torch.autograd.grad(density @ nodes, distances)
                helpers.assert_close(gradient, [1.0, 1.0], dtype)
