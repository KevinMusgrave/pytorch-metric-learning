import unittest

import torch

from pytorch_metric_learning.losses import NCALoss

from .. import TEST_DEVICE, TEST_DTYPES, helpers
from ..helpers import angle_to_coord


class TestNCALoss(unittest.TestCase):
    def test_nca_loss(self):
        softmax_scale = 10
        loss_func = NCALoss(softmax_scale=softmax_scale)

        for dtype in TEST_DTYPES:
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)

            loss = loss_func(embeddings, labels)
            loss.backward()

            pos_pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]
            neg_pairs = [
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 0),
                (2, 1),
                (2, 4),
                (3, 0),
                (3, 1),
                (3, 4),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 3),
            ]

            embeddings = torch.nn.functional.normalize(embeddings)
            correct_total = torch.zeros((), dtype=dtype, device=TEST_DEVICE)
            for a1, p in pos_pairs:
                anchor1, positive = embeddings[a1], embeddings[p]
                ap_dist = torch.sum((anchor1 - positive) ** 2)
                numerator = torch.exp(-ap_dist * softmax_scale)
                denominator = numerator.clone()
                for a2, n in neg_pairs:
                    if a2 == a1:
                        anchor2, negative = embeddings[a2], embeddings[n]
                        an_dist = torch.sum((anchor2 - negative) ** 2)
                        denominator += torch.exp(-an_dist * softmax_scale)

                correct_total += -torch.log(numerator / denominator)

            correct_total /= len(pos_pairs)
            rtol = helpers.rtol(dtype)
            self.assertTrue(torch.isclose(loss, correct_total, rtol=rtol))

    def test_zero_loss(self):
        loss_func = NCALoss(10)
        for dtype in TEST_DTYPES:
            for embedding_angles, labels in [([0, 20], [0, 0]), ([0], [0])]:
                embeddings = torch.tensor(
                    [angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(TEST_DEVICE)
                labels = torch.as_tensor(labels, dtype=torch.long)
                loss = loss_func(embeddings, labels)
                loss.backward()
                self.assertEqual(loss, 0)


def nca_reference(query, labels, reference, ref_labels, scale, exclude_self):
    query = torch.nn.functional.normalize(query, dim=1)
    reference = torch.nn.functional.normalize(reference, dim=1)
    logits = -scale * torch.cdist(query, reference) ** 2
    same = labels[:, None] == ref_labels[None, :]
    if exclude_self:
        eye = torch.eye(len(query), dtype=torch.bool, device=query.device)
        logits = logits.masked_fill(eye, -torch.inf)
        same = same & ~eye
    losses = torch.logsumexp(logits, 1) - torch.logsumexp(
        logits.masked_fill(~same, -torch.inf), 1
    )
    return losses[same.any(1)].mean()


class TestNCALogSpace(unittest.TestCase):
    def test_single_query_with_separate_references(self):
        labels = torch.tensor([0], device=TEST_DEVICE)
        ref_labels = torch.tensor([0, 1, 1], device=TEST_DEVICE)
        query = torch.tensor([[1.0, 0.2]])
        reference = torch.tensor([[0.9, 0.5], [1.0, -0.3], [-1.0, 0.1]])
        for dtype in TEST_DTYPES:
            loss = helpers.assert_matches_float64(
                lambda q, r: NCALoss()(q, labels, ref_emb=r, ref_labels=ref_labels),
                lambda q, r: nca_reference(q, labels, r, ref_labels, 1, False),
                [query, reference],
                dtype,
            )
            self.assertGreater(loss.item(), 0)

    def test_large_scale_keeps_hard_examples(self):
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        embeddings = torch.tensor([[1.0, 0.0], [-1.0, 0.05], [0.95, 0.3], [-0.9, -0.4]])
        for dtype in TEST_DTYPES:
            loss = helpers.assert_matches_float64(
                lambda x: NCALoss(softmax_scale=100)(x, labels),
                lambda x: nca_reference(x, labels, x, labels, 100, True),
                [embeddings],
                dtype,
                atol=1e-4,
            )
            self.assertGreater(loss.item(), 100)

    def test_rows_without_positives_are_excluded(self):
        embeddings = torch.randn(5, 3, device=TEST_DEVICE)
        labels = torch.tensor([0, 0, 1, 2, 3], device=TEST_DEVICE)
        loss = NCALoss()(embeddings, labels)
        expected = nca_reference(embeddings, labels, embeddings, labels, 1, True)
        torch.testing.assert_close(loss, expected)
