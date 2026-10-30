import unittest
from functools import partial

import torch

from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.reducers import MeanReducer

from .. import TEST_DEVICE, TEST_DTYPES, helpers
from ..helpers import angle_to_coord
from .utils import get_triplet_embeddings_with_ref


class TestTripletMarginLoss(unittest.TestCase):
    def test_default_margin_agrees_with_default_miner(self):
        embeddings = torch.randn(32, 8, device=TEST_DEVICE)
        labels = torch.arange(4, device=TEST_DEVICE).repeat_interleave(8)
        triplets = TripletMarginMiner()(embeddings, labels)
        terms = TripletMarginLoss(reducer=MeanReducer()).loss_terms(
            embeddings, labels, triplets
        )
        self.assertTrue(torch.all(terms["loss"]["losses"] > 0))
        self.assertGreater(len(triplets[0]), 0)
        all_terms = TripletMarginLoss().loss_terms(embeddings, labels)
        self.assertEqual(
            int(torch.count_nonzero(all_terms["loss"]["losses"])), len(triplets[0])
        )

    def test_triplet_margin_loss(self):
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(
                5,
                32,
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            embeddings = torch.nn.functional.normalize(embeddings)
            labels = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long).to(TEST_DEVICE)

            triplets = [
                (0, 1, 2),
                (0, 1, 3),
                (0, 1, 4),
                (1, 0, 2),
                (1, 0, 3),
                (1, 0, 4),
                (2, 3, 0),
                (2, 3, 1),
                (2, 3, 4),
                (3, 2, 0),
                (3, 2, 1),
                (3, 2, 4),
            ]

            self.helper(embeddings, labels, triplets, dtype)

    def test_triplet_margin_loss_with_ref(self):
        for dtype in TEST_DTYPES:
            (
                embeddings,
                labels,
                ref_emb,
                ref_labels,
                triplets,
            ) = get_triplet_embeddings_with_ref(dtype, TEST_DEVICE)
            self.helper(embeddings, labels, triplets, dtype, ref_emb, ref_labels)

    def helper(
        self, embeddings, labels, triplets, dtype, ref_emb=None, ref_labels=None
    ):
        margin = 0.2
        loss_funcA = TripletMarginLoss(margin=margin)
        loss_funcB = TripletMarginLoss(margin=margin, reducer=MeanReducer())
        loss_funcC = TripletMarginLoss(margin=margin, distance=CosineSimilarity())
        loss_funcD = TripletMarginLoss(
            margin=margin, reducer=MeanReducer(), distance=CosineSimilarity()
        )
        loss_funcE = TripletMarginLoss(margin=margin, smooth_loss=True)

        [lossA, lossB, lossC, lossD, lossE] = [
            x(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
            for x in [loss_funcA, loss_funcB, loss_funcC, loss_funcD, loss_funcE]
        ]

        correct_loss = torch.zeros((), dtype=dtype, device=TEST_DEVICE)
        correct_loss_cosine = torch.zeros((), dtype=dtype, device=TEST_DEVICE)
        correct_smooth_loss = torch.zeros((), dtype=dtype, device=TEST_DEVICE)
        num_non_zero_triplets = 0
        num_non_zero_triplets_cosine = 0
        for a, p, n in triplets:
            anchor = embeddings[a]
            if ref_emb is not None:
                positive, negative = ref_emb[p], ref_emb[n]
            else:
                positive, negative = embeddings[p], embeddings[n]
            ap_dist = torch.sqrt(torch.sum((anchor - positive) ** 2))
            an_dist = torch.sqrt(torch.sum((anchor - negative) ** 2))
            curr_loss = torch.relu(ap_dist - an_dist + margin)
            curr_loss_cosine = torch.relu(
                torch.sum(anchor * negative) - torch.sum(anchor * positive) + margin
            )
            correct_smooth_loss += torch.nn.functional.softplus(
                ap_dist - an_dist + margin
            )
            if curr_loss > 0:
                num_non_zero_triplets += 1
            if curr_loss_cosine > 0:
                num_non_zero_triplets_cosine += 1
            correct_loss += curr_loss
            correct_loss_cosine += curr_loss_cosine
        rtol = helpers.rtol(dtype)
        self.assertTrue(
            torch.isclose(lossA, correct_loss / num_non_zero_triplets, rtol=rtol)
        )
        self.assertTrue(torch.isclose(lossB, correct_loss / len(triplets), rtol=rtol))
        self.assertTrue(
            torch.isclose(
                lossC, correct_loss_cosine / num_non_zero_triplets_cosine, rtol=rtol
            )
        )
        self.assertTrue(
            torch.isclose(lossD, correct_loss_cosine / len(triplets), rtol=rtol)
        )
        self.assertTrue(
            torch.isclose(lossE, correct_smooth_loss / len(triplets), rtol=rtol)
        )

    def test_with_no_valid_triplets(self):
        loss_funcA = TripletMarginLoss(margin=0.2)
        loss_funcB = TripletMarginLoss(margin=0.2, reducer=MeanReducer())
        for dtype in TEST_DTYPES:
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long).to(TEST_DEVICE)
            lossA = loss_funcA(embeddings, labels)
            lossB = loss_funcB(embeddings, labels)
            self.assertEqual(lossA, 0)
            self.assertEqual(lossB, 0)

    def test_backward(self):
        labels = torch.tensor([0, 0, 1, 1, 2], device=TEST_DEVICE)
        coordinates = torch.tensor([angle_to_coord(a) for a in [0, 20, 40, 60, 80]])
        cases = (
            (TripletMarginLoss(margin=0.2), {"margin": 0.2}),
            (
                TripletMarginLoss(margin=0.2, reducer=MeanReducer()),
                {"margin": 0.2, "nonzero_mean": False},
            ),
            (
                TripletMarginLoss(smooth_loss=True),
                {"margin": 0.2, "smooth": True, "nonzero_mean": False},
            ),
        )
        for loss_func, options in cases:
            for dtype in TEST_DTYPES:
                with self.subTest(options=options, dtype=dtype):
                    loss = helpers.assert_matches_float64(
                        partial(loss_func, labels=labels),
                        partial(triplet_reference, labels=labels, **options),
                        [coordinates],
                        dtype,
                    )
                    self.assertGreater(loss.item(), 0)

    def test_swap_uses_closer_positive_negative_distance(self):
        torch.manual_seed(11)
        labels = torch.tensor([0, 0, 1, 1, 2, 0], device=TEST_DEVICE)
        embeddings = torch.randn(6, 3)
        reference_embeddings = torch.randn(7, 3)
        ref_labels = torch.tensor([0, 1, 2, 0, 1, 2, 1], device=TEST_DEVICE)
        loss_func = TripletMarginLoss(margin=0.3, swap=True)
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype, ref=False):
                swapped = helpers.assert_matches_float64(
                    lambda x: loss_func(x, labels),
                    lambda x: triplet_reference(x, labels, margin=0.3, swap=True),
                    [embeddings],
                    dtype,
                )
                plain = triplet_reference(
                    embeddings.to(TEST_DEVICE, torch.float64), labels, margin=0.3
                )
                self.assertNotAlmostEqual(swapped.item(), plain.item(), places=3)
            with self.subTest(dtype=dtype, ref=True):
                helpers.assert_matches_float64(
                    lambda x, r: loss_func(x, labels, ref_emb=r, ref_labels=ref_labels),
                    lambda x, r: triplet_reference(
                        x, labels, margin=0.3, swap=True, ref=r, ref_labels=ref_labels
                    ),
                    [embeddings, reference_embeddings],
                    dtype,
                )


def triplet_reference(
    embeddings,
    labels,
    margin,
    swap=False,
    smooth=False,
    nonzero_mean=True,
    ref=None,
    ref_labels=None,
):
    x = torch.nn.functional.normalize(embeddings, dim=1)
    r = x if ref is None else torch.nn.functional.normalize(ref, dim=1)
    r_labels = labels if ref_labels is None else ref_labels

    def distance(u, v):
        return torch.sqrt(torch.sum((u - v) ** 2))

    losses = []
    for a in range(len(x)):
        for p in range(len(r)):
            if r_labels[p] != labels[a] or (ref is None and p == a):
                continue
            for n in range(len(r)):
                if r_labels[n] == labels[a]:
                    continue
                an = distance(x[a], r[n])
                if swap:
                    an = torch.minimum(an, distance(r[p], r[n]))
                violation = distance(x[a], r[p]) - an + margin
                losses.append(
                    torch.nn.functional.softplus(violation)
                    if smooth
                    else torch.relu(violation)
                )
    stacked = torch.stack(losses)
    if nonzero_mean:
        return stacked.sum() / (stacked > 0).sum()
    return stacked.mean()
