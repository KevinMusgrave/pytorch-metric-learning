import unittest

import torch
import torch.nn.functional as F

from pytorch_metric_learning.distances import (
    CosineSimilarity,
    DotProductSimilarity,
    LpDistance,
)
from pytorch_metric_learning.losses import (
    ContrastiveLoss,
    MultipleLosses,
    ThresholdConsistentMarginLoss,
)

from .. import TEST_DEVICE, TEST_DTYPES, helpers


class TestThresholdConsistentMarginLoss(unittest.TestCase):
    def test_tcm_loss(self):
        torch.manual_seed(3459)
        for dtype in TEST_DTYPES:
            loss_func = MultipleLosses(
                losses={
                    "contrastive": ContrastiveLoss(
                        distance=CosineSimilarity(),
                        pos_margin=0.9,
                        neg_margin=0.4,
                    ),
                    "tcm": ThresholdConsistentMarginLoss(),
                }
            )
            embs = torch.tensor(
                [
                    [0.00, 1.00],
                    [0.43, 0.90],
                    [1.00, 0.00],
                    [0.50, 0.50],
                ],
                device=TEST_DEVICE,
                dtype=dtype,
                requires_grad=True,
            )
            labels = torch.tensor([0, 0, 1, 1]).to(TEST_DEVICE)

            contrastive_loss = 0.4866
            margin_plus, margin_minus = 0.9, 0.5
            positive_sim_2_3 = 0.7071
            negative_sim_0_3, negative_sim_1_3 = 0.7071, 0.9429
            tcm_positive_part = margin_plus - positive_sim_2_3
            tcm_negative_part = (
                (negative_sim_0_3 - margin_minus) + (negative_sim_1_3 - margin_minus)
            ) / 2
            correct_loss = torch.tensor(
                contrastive_loss + tcm_positive_part + tcm_negative_part
            ).to(dtype)

            with torch.no_grad():
                res = loss_func.forward(embs, labels)
                rtol = helpers.rtol(dtype)
                atol = 1e-4
                self.assertTrue(torch.isclose(res, correct_loss, rtol=rtol, atol=atol))


class TestThresholdConsistentMarginReference(unittest.TestCase):
    def test_no_margin_violations_produce_differentiable_zero(self):
        loss_func = ThresholdConsistentMarginLoss()
        for dtype in TEST_DTYPES:
            for labels in ([0, 0, 1, 1], [0, 0, 0, 0], [0, 1, 2, 3]):
                with self.subTest(dtype=dtype, labels=labels):
                    embeddings = torch.eye(4, device=TEST_DEVICE, dtype=dtype)[
                        labels
                    ].requires_grad_()
                    loss = loss_func(
                        embeddings, torch.tensor(labels, device=TEST_DEVICE)
                    )
                    self.assertEqual(loss.item(), 0)
                    loss.backward()
                    torch.testing.assert_close(
                        embeddings.grad, torch.zeros_like(embeddings)
                    )

    def test_reference_embeddings_are_used(self):
        torch.manual_seed(7)
        query = torch.randn(4, 3, dtype=torch.float64, device=TEST_DEVICE)
        reference = torch.randn(6, 3, dtype=torch.float64, device=TEST_DEVICE)
        labels = torch.tensor([0, 1, 0, 1]).to(TEST_DEVICE)
        ref_labels = torch.tensor([0, 0, 1, 1, 2, 1]).to(TEST_DEVICE)
        loss = ThresholdConsistentMarginLoss(margin_plus=0.5, margin_minus=-0.5)(
            query, labels, ref_emb=reference, ref_labels=ref_labels
        )
        sims = F.normalize(query, dim=1) @ F.normalize(reference, dim=1).T
        same = labels[:, None] == ref_labels[None, :]
        pos, neg = sims[same], sims[~same]
        pos_terms = (0.5 - pos)[pos <= 0.5]
        neg_terms = (neg + 0.5)[neg >= -0.5]
        expected = pos_terms.mean() + neg_terms.mean()
        torch.testing.assert_close(loss, expected)

    def test_configured_similarity_is_used(self):
        torch.manual_seed(5)
        embeddings = 0.6 * torch.randn(6, 3)
        labels = torch.tensor([0, 0, 1, 1, 2, 0], device=TEST_DEVICE)
        loss_func = ThresholdConsistentMarginLoss(
            margin_plus=0.4,
            margin_minus=-0.1,
            distance=DotProductSimilarity(normalize_embeddings=False),
        )

        def reference(x):
            sims = x @ x.T
            same = labels[:, None] == labels[None, :]
            others = ~torch.eye(len(labels), dtype=torch.bool, device=x.device)
            pos, neg = sims[same & others], sims[~same]
            pos_terms = (0.4 - pos)[pos <= 0.4]
            neg_terms = (neg + 0.1)[neg >= -0.1]
            return pos_terms.mean() + neg_terms.mean()

        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                loss = helpers.assert_matches_float64(
                    lambda x: loss_func(x, labels), reference, [embeddings], dtype
                )
                cosine_loss = ThresholdConsistentMarginLoss(
                    margin_plus=0.4, margin_minus=-0.1
                )(embeddings.to(TEST_DEVICE, dtype), labels)
                self.assertNotAlmostEqual(loss.item(), cosine_loss.item(), places=3)

    def test_distance_must_be_a_similarity(self):
        with self.assertRaises(TypeError):
            ThresholdConsistentMarginLoss(distance=LpDistance())

    def test_float16_accumulates_large_batches_in_float32(self):
        generator = torch.Generator().manual_seed(0)
        embeddings = torch.randn(512, 32, generator=generator).to(TEST_DEVICE)
        labels = torch.arange(512, device=TEST_DEVICE) % 2
        loss_func = ThresholdConsistentMarginLoss()
        expected = loss_func(embeddings.double(), labels)
        actual = loss_func(embeddings.half(), labels)
        self.assertEqual(actual.dtype, torch.float16)
        helpers.assert_close(actual, expected, torch.float16)
