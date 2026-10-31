import math
import unittest

import torch

from pytorch_metric_learning import distances, losses, miners

from .. import TEST_DEVICE, TEST_DTYPES, helpers


def unit(x):
    return x / x.norm(dim=1, keepdim=True)


def triplets_of(labels):
    rows = range(len(labels))
    return [
        (a, p, n)
        for a in rows
        for p in rows
        for n in rows
        if a != p and labels[a] == labels[p] and labels[a] != labels[n]
    ]


def equation_13(embeddings, labels, temperature, weight=1.0, threshold=None):
    scores = unit(embeddings) @ unit(embeddings).T
    terms = []
    for a, p, n in triplets_of(labels.tolist()):
        s_ap, s_an = scores[a, p], scores[a, n]
        hard = s_an > s_ap or (threshold is not None and s_an > threshold)
        if hard:
            terms.append(weight * s_an)
        else:
            numerator = torch.exp(s_ap / temperature)
            denominator = numerator + torch.exp(s_an / temperature)
            terms.append(-torch.log(numerator / denominator))
    return torch.stack(terms).mean()


def planar(*angles, dtype=torch.float64):
    return torch.tensor(
        [helpers.angle_to_coord(angle, normalized=True) for angle in angles],
        dtype=dtype,
        device=TEST_DEVICE,
    )


class TestSelectivelyContrastiveTripletLoss(unittest.TestCase):
    def batch(self, dtype=torch.float64):
        generator = torch.Generator().manual_seed(4)
        embeddings = torch.randn(9, 2, generator=generator, dtype=torch.float64)
        labels = (torch.arange(9) % 3).to(TEST_DEVICE)
        return embeddings.to(TEST_DEVICE, dtype), labels

    def test_equation_13_on_every_triplet(self):
        for dtype in TEST_DTYPES:
            for temperature in (0.1, 1.0):
                with self.subTest(dtype=dtype, temperature=temperature):
                    embeddings, labels = self.batch(dtype)
                    loss_func = losses.SelectivelyContrastiveTripletLoss(
                        temperature=temperature
                    )
                    helpers.assert_matches_float64(
                        lambda x, f=loss_func, y=labels: f(x, y),
                        lambda x, y=labels, t=temperature: equation_13(x, y, t),
                        (embeddings,),
                        dtype,
                    )

    def test_batch_mixes_both_cases(self):
        embeddings, labels = self.batch()
        scores = unit(embeddings) @ unit(embeddings).T
        hard = [scores[a, n] > scores[a, p] for a, p, n in triplets_of(labels.tolist())]
        self.assertTrue(any(hard))
        self.assertFalse(all(hard))

    def test_hard_triplet_value(self):
        embeddings = planar(0, 60, 30)
        labels = torch.tensor([0, 0, 1], device=TEST_DEVICE)
        triplet = tuple(torch.tensor([x], device=TEST_DEVICE) for x in (0, 1, 2))
        for weight in (1.0, 2.5):
            with self.subTest(weight=weight):
                loss = losses.SelectivelyContrastiveTripletLoss(hard_weight=weight)
                self.assertAlmostEqual(
                    loss(embeddings, labels, triplet).item(),
                    weight * math.cos(math.radians(30)),
                )

    def test_easy_triplet_value(self):
        embeddings = planar(0, 30, 60)
        labels = torch.tensor([0, 0, 1], device=TEST_DEVICE)
        triplet = tuple(torch.tensor([x], device=TEST_DEVICE) for x in (0, 1, 2))
        s_ap, s_an = math.cos(math.radians(30)), math.cos(math.radians(60))
        expected = -math.log(
            math.exp(s_ap / 0.5) / (math.exp(s_ap / 0.5) + math.exp(s_an / 0.5))
        )
        loss = losses.SelectivelyContrastiveTripletLoss(temperature=0.5)
        self.assertAlmostEqual(loss(embeddings, labels, triplet).item(), expected)

    def test_hard_triplet_sends_no_gradient_to_the_positive(self):
        embeddings = planar(0, 60, 30).requires_grad_()
        labels = torch.tensor([0, 0, 1], device=TEST_DEVICE)
        triplet = tuple(torch.tensor([x], device=TEST_DEVICE) for x in (0, 1, 2))
        losses.SelectivelyContrastiveTripletLoss()(
            embeddings, labels, triplet
        ).backward()
        gradient = embeddings.grad
        assert gradient is not None
        self.assertEqual(gradient[1].abs().sum().item(), 0)
        self.assertGreater(gradient[0].abs().sum().item(), 0)
        self.assertGreater(gradient[2].abs().sum().item(), 0)

    def test_hard_step_lowers_negative_similarity(self):
        embeddings = planar(0, 60, 30).requires_grad_()
        labels = torch.tensor([0, 0, 1], device=TEST_DEVICE)
        triplet = tuple(torch.tensor([x], device=TEST_DEVICE) for x in (0, 1, 2))
        loss_func = losses.SelectivelyContrastiveTripletLoss()
        before = loss_func(embeddings, labels, triplet)
        before.backward()
        gradient = embeddings.grad
        assert gradient is not None
        moved = embeddings.detach() - 0.01 * gradient
        self.assertLess(loss_func(moved, labels, triplet).item(), before.item())

    def test_exact_tie_takes_the_triplet_case(self):
        embeddings = planar(0, 30, -30)
        labels = torch.tensor([0, 0, 1], device=TEST_DEVICE)
        triplet = tuple(torch.tensor([x], device=TEST_DEVICE) for x in (0, 1, 2))
        loss_func = losses.SelectivelyContrastiveTripletLoss()
        distance = loss_func.distance
        mat = distance(embeddings)
        mat = mat.clone()
        mat[0, 2] = mat[0, 1]
        loss = loss_func(embeddings, labels, triplet, mat=mat)
        self.assertAlmostEqual(loss.item(), math.log(2))

    def test_threshold_matches_the_authors_rule(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                embeddings, labels = self.batch(dtype)
                loss_func = losses.SelectivelyContrastiveTripletLoss(hard_threshold=0.1)
                helpers.assert_matches_float64(
                    lambda x, f=loss_func, y=labels: f(x, y),
                    lambda x, y=labels: equation_13(x, y, 0.1, threshold=0.1),
                    (embeddings,),
                    dtype,
                )

    def test_threshold_changes_the_case_of_easy_triplets(self):
        embeddings = planar(0, 10, 20)
        labels = torch.tensor([0, 0, 1], device=TEST_DEVICE)
        triplet = tuple(torch.tensor([x], device=TEST_DEVICE) for x in (0, 1, 2))
        loss_func = losses.SelectivelyContrastiveTripletLoss(hard_threshold=0.8)
        self.assertAlmostEqual(
            loss_func(embeddings, labels, triplet).item(), math.cos(math.radians(20))
        )

    def test_miner_selects_one_triplet_per_anchor(self):
        embeddings, labels = self.batch()
        miner = miners.BatchEasyHardMiner(pos_strategy="easy", neg_strategy="hard")
        pairs = miner(embeddings, labels)
        terms = losses.SelectivelyContrastiveTripletLoss().loss_terms(
            embeddings, labels, pairs
        )
        self.assertEqual(len(terms["loss"]["losses"]), len(embeddings))

    def test_separate_reference_set(self):
        embeddings, labels = self.batch()
        reference = embeddings.flip(0)
        ref_labels = labels.flip(0)
        loss = losses.SelectivelyContrastiveTripletLoss(temperature=0.5)(
            embeddings, labels, ref_emb=reference, ref_labels=ref_labels
        )
        scores = unit(embeddings) @ unit(reference).T
        terms = []
        for a in range(9):
            for p in range(9):
                for n in range(9):
                    if labels[a] == ref_labels[p] and labels[a] != ref_labels[n]:
                        s_ap, s_an = scores[a, p], scores[a, n]
                        if s_an > s_ap:
                            terms.append(s_an)
                        else:
                            terms.append(
                                torch.nn.functional.softplus((s_an - s_ap) / 0.5)
                            )
        self.assertAlmostEqual(loss.item(), torch.stack(terms).mean().item())

    def test_no_triplets_give_a_differentiable_zero(self):
        embeddings = planar(0, 30, 60).requires_grad_()
        labels = torch.arange(3, device=TEST_DEVICE)
        loss = losses.SelectivelyContrastiveTripletLoss()(embeddings, labels)
        self.assertEqual(loss.item(), 0)
        loss.backward()

    def test_invalid_settings(self):
        for temperature in (0, -1):
            with (
                self.subTest(temperature=temperature),
                self.assertRaisesRegex(ValueError, "temperature"),
            ):
                losses.SelectivelyContrastiveTripletLoss(temperature=temperature)
        with self.assertRaisesRegex(ValueError, "requires a similarity"):
            losses.SelectivelyContrastiveTripletLoss(distance=distances.LpDistance())
