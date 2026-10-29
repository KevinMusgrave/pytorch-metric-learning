import unittest
from functools import partial

import torch

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import NTXentLoss, SupConLoss
from pytorch_metric_learning.reducers import (
    AvgNonZeroReducer,
    MeanReducer,
    PerAnchorReducer,
)

from .. import TEST_DEVICE, TEST_DTYPES, helpers
from ..helpers import angle_to_coord
from .utils import get_pair_embeddings_with_ref


class TestNTXentLoss(unittest.TestCase):
    def test_ntxent_loss(self):
        torch.manual_seed(5430)
        for dtype in TEST_DTYPES:
            embedding_angles = [0, 10, 20, 50, 60, 80]
            embeddings = torch.tensor(
                [angle_to_coord(a, normalized=True) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)

            labels = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long).to(TEST_DEVICE)
            pos_pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (3, 4), (4, 3)]
            neg_pairs = [
                (0, 3),
                (0, 4),
                (0, 5),
                (1, 3),
                (1, 4),
                (1, 5),
                (2, 3),
                (2, 4),
                (2, 5),
                (3, 0),
                (3, 1),
                (3, 2),
                (3, 5),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 5),
                (5, 0),
                (5, 1),
                (5, 2),
                (5, 3),
                (5, 4),
            ]

            self.helper(embeddings, labels, pos_pairs, neg_pairs, dtype)

    def test_ntxent_loss_with_ref(self):
        torch.manual_seed(139)
        for dtype in TEST_DTYPES:
            (
                embeddings,
                labels,
                ref_emb,
                ref_labels,
                pos_pairs,
                neg_pairs,
            ) = get_pair_embeddings_with_ref(dtype, TEST_DEVICE)
            self.helper(
                embeddings, labels, pos_pairs, neg_pairs, dtype, ref_emb, ref_labels
            )

    def helper(
        self,
        embeddings,
        labels,
        pos_pairs,
        neg_pairs,
        dtype,
        ref_emb=None,
        ref_labels=None,
    ):
        temperature = 0.1
        loss_funcA = NTXentLoss(temperature=temperature)
        loss_funcB = NTXentLoss(temperature=temperature, distance=LpDistance())
        loss_funcC = NTXentLoss(
            temperature=temperature, reducer=PerAnchorReducer(AvgNonZeroReducer())
        )
        loss_funcD = SupConLoss(temperature=temperature)
        loss_funcE = SupConLoss(temperature=temperature, distance=LpDistance())

        obtained_losses = [
            x(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
            for x in [loss_funcA, loss_funcB, loss_funcC, loss_funcD, loss_funcE]
        ]

        embeddings = torch.nn.functional.normalize(embeddings.detach().double())
        if ref_emb is not None:
            ref_emb = torch.nn.functional.normalize(ref_emb.detach().double())
        loss_len = 4 if ref_emb is not None else 5
        total_lossA, total_lossB, total_lossC, total_lossD, total_lossE = (
            torch.zeros((), device=TEST_DEVICE, dtype=torch.float64),
            torch.zeros((), device=TEST_DEVICE, dtype=torch.float64),
            torch.zeros(loss_len, device=TEST_DEVICE, dtype=torch.float64),
            torch.zeros(loss_len, device=TEST_DEVICE, dtype=torch.float64),
            torch.zeros(loss_len, device=TEST_DEVICE, dtype=torch.float64),
        )
        for a1, p in pos_pairs:
            anchor = embeddings[a1]
            positive = embeddings[p] if ref_emb is None else ref_emb[p]
            numeratorA = torch.exp(torch.matmul(anchor, positive) / temperature)
            numeratorB = torch.exp(
                -torch.sqrt(torch.sum((anchor - positive) ** 2)) / temperature
            )
            denominatorA = numeratorA.clone()
            denominatorB = numeratorB.clone()
            denominatorD = 0
            denominatorE = 0
            for a2, n in pos_pairs + neg_pairs:
                if a2 == a1:
                    negative = embeddings[n] if ref_emb is None else ref_emb[n]
                    curr_denomD = torch.exp(
                        torch.matmul(anchor, negative) / temperature
                    )
                    curr_denomE = torch.exp(
                        -torch.sqrt(torch.sum((anchor - negative) ** 2)) / temperature
                    )
                    denominatorD += curr_denomD
                    denominatorE += curr_denomE
                    if (a2, n) not in pos_pairs:
                        denominatorA += curr_denomD
                        denominatorB += curr_denomE
                else:
                    continue

            curr_lossA = -torch.log(numeratorA / denominatorA)
            curr_lossB = -torch.log(numeratorB / denominatorB)
            curr_lossD = -torch.log(numeratorA / denominatorD)
            curr_lossE = -torch.log(numeratorB / denominatorE)
            total_lossA += curr_lossA
            total_lossB += curr_lossB
            total_lossC[a1] += curr_lossA
            total_lossD[a1] += curr_lossD
            total_lossE[a1] += curr_lossE

        total_lossA /= len(pos_pairs)
        total_lossB /= len(pos_pairs)
        if ref_emb is not None:
            pos_pair_per_anchor = torch.tensor([1, 1, 1, 1])
        else:
            pos_pair_per_anchor = torch.tensor([2, 2, 2, 1, 1])
        pos_pair_per_anchor = pos_pair_per_anchor.to(TEST_DEVICE, torch.float64)
        total_lossC, total_lossD, total_lossE = [
            torch.mean(x / pos_pair_per_anchor)
            for x in [total_lossC, total_lossD, total_lossE]
        ]

        expected = [total_lossA, total_lossB, total_lossC, total_lossD, total_lossE]
        for obtained, correct in zip(obtained_losses, expected, strict=True):
            helpers.assert_close(obtained, correct, dtype, atol=1e-8)

    def test_with_no_valid_pairs(self):
        all_embedding_angles = [[0], [0, 10, 20], [0, 40, 60]]
        all_labels = [
            torch.tensor([0], dtype=torch.long, device=TEST_DEVICE),
            torch.tensor([0, 0, 0], dtype=torch.long, device=TEST_DEVICE),
            torch.tensor([1, 2, 3], dtype=torch.long, device=TEST_DEVICE),
        ]
        temperature = 0.1
        for loss_class in [NTXentLoss, SupConLoss]:
            loss_funcA = loss_class(temperature)
            loss_funcB = loss_class(temperature, distance=LpDistance())
            for loss_func in [loss_funcA, loss_funcB]:
                for dtype in TEST_DTYPES:
                    for embedding_angles, labels in zip(
                        all_embedding_angles, all_labels, strict=True
                    ):
                        embeddings = torch.tensor(
                            [angle_to_coord(a) for a in embedding_angles],
                            requires_grad=True,
                            dtype=dtype,
                        ).to(TEST_DEVICE)
                        loss = loss_func(embeddings, labels)
                        loss.backward()
                        expected = 0
                        if loss_class is SupConLoss and len(labels) > 1 == len(
                            labels.unique()
                        ):
                            expected = supcon_without_negatives(
                                embeddings, loss_func.distance, temperature
                            )
                        torch.testing.assert_close(
                            loss.detach(),
                            torch.as_tensor(expected, dtype=dtype, device=TEST_DEVICE),
                            rtol=helpers.rtol(dtype),
                            atol=1e-6,
                        )

    def test_backward(self):
        temperature = 0.1
        labels = torch.tensor([0, 0, 1, 1, 2], device=TEST_DEVICE)
        coordinates = torch.tensor([angle_to_coord(a) for a in [0, 20, 40, 60, 80]])
        for loss_class, reference in (
            (NTXentLoss, ntxent_logits_reference),
            (SupConLoss, supcon_logits_reference),
        ):
            for distance, similarity in (
                (None, cosine_logits),
                (LpDistance(), negative_euclidean_logits),
            ):
                loss_func = loss_class(temperature, distance=distance)
                for dtype in TEST_DTYPES:
                    with self.subTest(loss=loss_class.__name__, dtype=dtype):
                        helpers.assert_matches_float64(
                            partial(loss_func, labels=labels),
                            partial(
                                logits_reference,
                                loss_reference=reference,
                                similarity=similarity,
                                temperature=temperature,
                                labels=labels.tolist(),
                            ),
                            [coordinates],
                            dtype,
                        )

    def test_supcon_excludes_anchors_without_positives(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 3], device=TEST_DEVICE)
        torch.manual_seed(9)
        embeddings = torch.randn(6, 3)
        loss_func = SupConLoss(0.2, reducer=MeanReducer())
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                helpers.assert_matches_float64(
                    lambda x: loss_func(x, labels),
                    lambda x: supcon_logits_reference(
                        cosine_logits(x) / 0.2, labels.tolist()
                    ),
                    [embeddings],
                    dtype,
                )


def logits_reference(embeddings, loss_reference, similarity, temperature, labels):
    return loss_reference(similarity(embeddings) / temperature, labels)


def cosine_logits(embeddings):
    normalized = torch.nn.functional.normalize(embeddings, dim=1)
    return normalized @ normalized.T


def negative_euclidean_logits(embeddings):
    normalized = torch.nn.functional.normalize(embeddings, dim=1)
    rows = len(normalized)
    return -torch.stack(
        [
            torch.stack(
                [
                    torch.sqrt(torch.sum((normalized[i] - normalized[j]) ** 2))
                    if i != j
                    else normalized.new_zeros(())
                    for j in range(rows)
                ]
            )
            for i in range(rows)
        ]
    )


def ntxent_logits_reference(logits, labels):
    losses = []
    for a, label in enumerate(labels):
        negatives = [n for n, other in enumerate(labels) if other != label]
        for p, other in enumerate(labels):
            if p != a and other == label:
                terms = torch.stack([logits[a, p], *[logits[a, n] for n in negatives]])
                losses.append(torch.logsumexp(terms, 0) - logits[a, p])
    return torch.stack(losses).mean()


def supcon_logits_reference(logits, labels):
    losses = []
    for a, label in enumerate(labels):
        others = [k for k in range(len(labels)) if k != a]
        positives = [p for p in others if labels[p] == label]
        if not positives:
            continue
        log_denominator = torch.logsumexp(logits[a, others], 0)
        losses.append(
            -torch.stack([logits[a, p] - log_denominator for p in positives]).mean()
        )
    return torch.stack(losses).mean()


def supcon_without_negatives(embeddings, distance, temperature):
    mat = distance(embeddings.detach().double())
    logits = (mat if distance.is_inverted else -mat) / temperature
    total, count = 0.0, 0
    for i in range(len(logits)):
        others = [j for j in range(len(logits)) if j != i]
        denominator = torch.logsumexp(logits[i, others], dim=0)
        loss = -sum(logits[i, j] - denominator for j in others) / len(others)
        if loss > 0:
            total, count = total + loss, count + 1
    return float(total / count) if count else 0.0


def ntxent_reference(embeddings, labels, temperature):
    normalized = torch.nn.functional.normalize(embeddings, dim=1)
    logits = normalized @ normalized.T / temperature
    losses = []
    for anchor in range(len(labels)):
        negatives = logits[anchor, labels != labels[anchor]]
        for positive in range(len(labels)):
            if positive != anchor and labels[positive] == labels[anchor]:
                pos = logits[anchor, positive]
                losses.append(
                    torch.logsumexp(torch.cat([pos[None], negatives]), 0) - pos
                )
    return torch.stack(losses).mean()


class TestNTXentPrecision(unittest.TestCase):
    def test_hard_positive_matches_float64_in_low_precision(self):
        labels = torch.tensor([0, 0, 1, 2], device=TEST_DEVICE)
        embeddings = torch.tensor([[1.0, 0.0], [-1.0, 0.02], [1.0, 0.05], [1.0, -0.05]])
        for dtype in {*TEST_DTYPES, torch.float16}:
            loss = helpers.assert_matches_float64(
                lambda x: NTXentLoss()(x, labels),
                lambda x: ntxent_reference(x, labels, 0.07),
                [embeddings],
                dtype,
                atol=1e-5,
            )
            self.assertGreater(loss.item(), 14)

    def test_anchor_negatives_are_grouped_without_cartesian_products(self):
        labels = torch.randint(0, 4, (24,), device=TEST_DEVICE)
        embeddings = torch.randn(24, 5)
        for dtype in TEST_DTYPES:
            helpers.assert_matches_float64(
                lambda x: NTXentLoss(temperature=0.2)(x, labels),
                lambda x: ntxent_reference(x, labels, 0.2),
                [embeddings],
                dtype,
            )
