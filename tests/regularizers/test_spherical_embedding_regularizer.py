import unittest

import torch

from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.regularizers import (
    CenterInvariantRegularizer,
    LpRegularizer,
    SphericalEmbeddingRegularizer,
)

from .. import TEST_DEVICE, TEST_DTYPES, helpers


def equation_17(x):
    norms = x.pow(2).sum(dim=1).sqrt()
    return (norms - norms.sum() / len(x)).pow(2).sum() / len(x)


def embeddings(dtype=torch.float64):
    generator = torch.Generator().manual_seed(1)
    rows = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    scale = torch.linspace(0.5, 3, 7, dtype=torch.float64).unsqueeze(1)
    return (rows * scale).to(TEST_DEVICE, dtype)


class TestSphericalEmbeddingRegularizer(unittest.TestCase):
    def test_equation_17_value_and_gradient(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                helpers.assert_matches_float64(
                    SphericalEmbeddingRegularizer(),
                    equation_17,
                    (embeddings(dtype),),
                    dtype,
                )

    def test_hand_computed_value(self):
        x = torch.tensor([[3.0, 4.0], [0.0, 1.0]], device=TEST_DEVICE)
        self.assertAlmostEqual(SphericalEmbeddingRegularizer()(x).item(), 4.0)

    def test_gradient_is_equation_18(self):
        x = embeddings().requires_grad_()
        SphericalEmbeddingRegularizer()(x).backward()
        rows = x.detach()
        norms = rows.norm(dim=1, keepdim=True)
        expected = 2 / len(rows) * (norms - norms.mean()) * rows / norms
        assert x.grad is not None
        torch.testing.assert_close(x.grad, expected)

    def test_equal_norms_are_not_penalized_at_any_radius(self):
        directions = torch.nn.functional.normalize(embeddings(), dim=1)
        for radius in (0.1, 1.0, 25.0):
            with self.subTest(radius=radius):
                x = (directions * radius).requires_grad_()
                loss = SphericalEmbeddingRegularizer()(x)
                self.assertAlmostEqual(loss.item(), 0)
                loss.backward()
                assert x.grad is not None
                self.assertAlmostEqual(x.grad.abs().max().item(), 0)

    def test_gradient_moves_norms_toward_the_mean(self):
        x = embeddings().requires_grad_()
        loss = SphericalEmbeddingRegularizer()(x)
        loss.backward()
        assert x.grad is not None
        moved = x.detach() - 0.5 * x.grad
        self.assertLess(moved.norm(dim=1).std(), x.detach().norm(dim=1).std())

    def test_step_keeps_directions(self):
        x = embeddings().requires_grad_()
        SphericalEmbeddingRegularizer()(x).backward()
        assert x.grad is not None
        moved = x.detach() - 0.5 * x.grad
        cosine = torch.nn.functional.cosine_similarity(moved, x.detach())
        torch.testing.assert_close(cosine, torch.ones_like(cosine))

    def test_differs_from_center_invariant_regularizer(self):
        first = embeddings().requires_grad_()
        second = (embeddings() * 2).requires_grad_()
        sec = SphericalEmbeddingRegularizer()
        center_invariant = CenterInvariantRegularizer()
        sec_ratio = sec(second) / sec(first)
        center_ratio = center_invariant(second) / center_invariant(first)
        self.assertAlmostEqual(sec_ratio.item(), 4)
        self.assertAlmostEqual(center_ratio.item(), 16)
        x = embeddings().requires_grad_()
        sec_gradient = torch.autograd.grad(sec(x), x)[0]
        center_gradient = torch.autograd.grad(center_invariant(x), x)[0]
        ratios = (center_gradient / sec_gradient).flatten()
        self.assertGreater(ratios.std().item(), 1)

    def test_paper_l2_reg_is_the_zero_radius_case(self):
        x = embeddings()
        l2_reg = LpRegularizer(power=2)(x)
        norms = x.norm(dim=1)
        self.assertAlmostEqual(l2_reg.item(), norms.pow(2).mean().item())
        self.assertGreater(l2_reg.item(), SphericalEmbeddingRegularizer()(x).item())

    def test_loss_adds_weighted_penalty_on_unnormalized_embeddings(self):
        x = embeddings().requires_grad_()
        labels = (torch.arange(7) % 3).to(TEST_DEVICE)
        plain = losses.ContrastiveLoss()
        regularized = losses.ContrastiveLoss(
            embedding_regularizer=SphericalEmbeddingRegularizer(),
            embedding_reg_weight=0.5,
        )
        total = regularized(x, labels)
        expected = plain(x, labels) + 0.5 * equation_17(x)
        torch.testing.assert_close(total, expected)
        terms = regularized.loss_terms(x, labels)
        self.assertIn("embedding_reg_loss", terms)
        self.assertGreater(terms["embedding_reg_loss"]["losses"].item(), 0)

    def test_normalized_embeddings_give_no_penalty(self):
        x = torch.nn.functional.normalize(embeddings(), dim=1)
        self.assertAlmostEqual(SphericalEmbeddingRegularizer()(x).item(), 0)

    def test_zero_row_keeps_gradients_finite(self):
        x = torch.tensor(
            [[0.0, 0.0], [3.0, 4.0]], dtype=torch.float64, device=TEST_DEVICE
        ).requires_grad_()
        loss = SphericalEmbeddingRegularizer()(x)
        self.assertAlmostEqual(loss.item(), 6.25)
        loss.backward()
        assert x.grad is not None
        self.assertTrue(x.grad.isfinite().all())
        self.assertEqual(x.grad[0].abs().sum().item(), 0)

    def test_low_precision_sum_does_not_overflow(self):
        x = torch.full((64, 256), 200.0, dtype=torch.float16, device=TEST_DEVICE)
        x[0] = 180.0
        self.assertGreater(
            (x.double().norm(dim=1) - x.double().norm(dim=1).mean()).pow(2).max(),
            torch.finfo(torch.float16).max,
        )
        loss = SphericalEmbeddingRegularizer()(x)
        self.assertEqual(loss.dtype, torch.float16)
        self.assertTrue(loss.isfinite())
        torch.testing.assert_close(
            loss.double(), equation_17(x.double()), rtol=2e-3, atol=0
        )

    def test_reducer_option(self):
        x = embeddings()
        total = SphericalEmbeddingRegularizer(reducer=reducers.SumReducer())(x)
        self.assertAlmostEqual(total.item(), equation_17(x).item() * len(x))

    def test_empty_input_gives_a_zero(self):
        self.assertEqual(
            SphericalEmbeddingRegularizer()(
                torch.empty(0, 4, device=TEST_DEVICE)
            ).item(),
            0,
        )
