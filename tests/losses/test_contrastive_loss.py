import unittest
from functools import partial

import torch

from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.reducers import MeanReducer

from .. import TEST_DEVICE, TEST_DTYPES, helpers


def pair_losses(embeddings, pairs, loss_of_pair):
    return torch.stack([loss_of_pair(embeddings[a], embeddings[b]) for a, b in pairs])


def mean_of_non_zero(losses):
    return losses.sum() / (losses > 0).sum().clamp(min=1)


def contrastive_reference(
    embeddings, positives, negatives, measure, sign, pos_margin, neg_margin
):
    x = torch.nn.functional.normalize(embeddings, dim=1)
    pos = pair_losses(
        x, positives, lambda a, b: torch.relu(sign * (measure(a, b) - pos_margin))
    )
    neg = pair_losses(
        x, negatives, lambda a, b: torch.relu(sign * (neg_margin - measure(a, b)))
    )
    return mean_of_non_zero(pos) + mean_of_non_zero(neg)


class TestContrastiveLoss(unittest.TestCase):
    def test_contrastive_loss(self):
        loss_funcA = ContrastiveLoss(
            pos_margin=0.25, neg_margin=1.5, distance=LpDistance(power=2)
        )
        loss_funcB = ContrastiveLoss(
            pos_margin=1.5, neg_margin=0.6, distance=CosineSimilarity()
        )
        loss_funcC = ContrastiveLoss(
            pos_margin=0.25,
            neg_margin=1.5,
            distance=LpDistance(power=2),
            reducer=MeanReducer(),
        )
        loss_funcD = ContrastiveLoss(
            pos_margin=1.5,
            neg_margin=0.6,
            distance=CosineSimilarity(),
            reducer=MeanReducer(),
        )

        for dtype in TEST_DTYPES:
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor(
                [helpers.angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long).to(TEST_DEVICE)

            lossA = loss_funcA(embeddings, labels)
            lossB = loss_funcB(embeddings, labels)
            lossC = loss_funcC(embeddings, labels)
            lossD = loss_funcD(embeddings, labels)

            embeddings = torch.nn.functional.normalize(embeddings)

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

            lp_pos = pair_losses(
                embeddings,
                pos_pairs,
                lambda a, p: torch.relu(torch.sum((a - p) ** 2) - 0.25),
            )
            cos_pos = pair_losses(
                embeddings, pos_pairs, lambda a, p: torch.relu(1.5 - torch.matmul(a, p))
            )
            lp_neg = pair_losses(
                embeddings,
                neg_pairs,
                lambda a, n: torch.relu(1.5 - torch.sum((a - n) ** 2)),
            )
            cos_neg = pair_losses(
                embeddings, neg_pairs, lambda a, n: torch.relu(torch.matmul(a, n) - 0.6)
            )
            correct_losses = [
                mean_of_non_zero(lp_pos) + mean_of_non_zero(lp_neg),
                mean_of_non_zero(cos_pos) + mean_of_non_zero(cos_neg),
                lp_pos.mean() + lp_neg.mean(),
                cos_pos.mean() + cos_neg.mean(),
            ]

            rtol = helpers.rtol(dtype)
            self.assertTrue(torch.isclose(lossA, correct_losses[0], rtol=rtol))
            self.assertTrue(torch.isclose(lossB, correct_losses[1], rtol=rtol))
            self.assertTrue(torch.isclose(lossC, correct_losses[2], rtol=rtol))
            self.assertTrue(torch.isclose(lossD, correct_losses[3], rtol=rtol))

    def test_with_no_valid_pairs(self):
        loss_funcA = ContrastiveLoss()
        loss_funcB = ContrastiveLoss(distance=CosineSimilarity())
        for dtype in TEST_DTYPES:
            embedding_angles = [0]
            embeddings = torch.tensor(
                [helpers.angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.tensor([0], dtype=torch.long).to(TEST_DEVICE)
            lossA = loss_funcA(embeddings, labels)
            lossB = loss_funcB(embeddings, labels)
            self.assertEqual(lossA, 0)
            self.assertEqual(lossB, 0)

    def test_backward(self):
        labels = torch.tensor([0, 0, 1, 1, 2], device=TEST_DEVICE)
        pairs = [(a, b) for a in range(5) for b in range(5) if a != b]
        positives = [(a, b) for a, b in pairs if labels[a] == labels[b]]
        negatives = [(a, b) for a, b in pairs if labels[a] != labels[b]]
        angles = torch.tensor([0.0, 25.0, 50.0, 120.0, 200.0])
        coordinates = torch.stack(
            [2 * torch.cos(torch.deg2rad(angles)), torch.sin(torch.deg2rad(angles))],
            dim=1,
        )

        def euclidean(x, y):
            return torch.sqrt(torch.sum((x - y) ** 2))

        def cosine(x, y):
            return torch.dot(x, y)

        cases = (
            (ContrastiveLoss(pos_margin=0.1, neg_margin=1.2), euclidean, 1),
            (
                ContrastiveLoss(
                    pos_margin=0.95, neg_margin=0.2, distance=CosineSimilarity()
                ),
                cosine,
                -1,
            ),
        )
        for dtype in TEST_DTYPES:
            for loss_func, measure, sign in cases:
                with self.subTest(dtype=dtype, distance=type(loss_func.distance)):
                    loss = helpers.assert_matches_float64(
                        partial(loss_func, labels=labels),
                        partial(
                            contrastive_reference,
                            positives=positives,
                            negatives=negatives,
                            measure=measure,
                            sign=sign,
                            pos_margin=loss_func.pos_margin,
                            neg_margin=loss_func.neg_margin,
                        ),
                        [coordinates],
                        dtype,
                    )
                    self.assertGreater(loss.item(), 0)
