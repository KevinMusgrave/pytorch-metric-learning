import unittest

import torch

from pytorch_metric_learning import distances, losses, reducers

from .. import TEST_DEVICE, TEST_DTYPES, helpers


def equations_1_to_3(embeddings, targets, temperature=2.0, sign=-1, score=None):
    if score is None:
        score = torch.linalg.vector_norm(
            embeddings[:, None] - embeddings[None, :], dim=-1
        )
    similarity = sign * score
    rows = range(len(targets))
    anchors = []
    for i in rows:
        terms = []
        for j in rows:
            if j == i:
                continue
            gap = abs(targets[i] - targets[j])
            denominator = [
                torch.exp(similarity[i, k] / temperature)
                for k in rows
                if k != i and abs(targets[i] - targets[k]) >= gap
            ]
            numerator = torch.exp(similarity[i, j] / temperature)
            terms.append(-torch.log(numerator / torch.stack(denominator).sum()))
        anchors.append(torch.stack(terms).mean())
    return torch.stack(anchors).mean()


def batch(dtype=torch.float64, size=7):
    generator = torch.Generator().manual_seed(8)
    embeddings = torch.randn(size, 4, generator=generator, dtype=torch.float64)
    return embeddings.to(TEST_DEVICE, dtype)


def tensor(values, dtype=torch.float64):
    return torch.tensor(values, dtype=dtype, device=TEST_DEVICE)


CASES = {
    "distinct": [0.5, 3.0, 1.0, 7.0, 2.0, 9.0, 4.5],
    "tied-distances": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    "duplicate-targets": [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 8.0],
    "all-equal": [4.0] * 7,
}


class TestRankNContrastLoss(unittest.TestCase):
    def test_equations_with_and_without_ties(self):
        for name, values in CASES.items():
            for dtype in TEST_DTYPES:
                for temperature in (0.5, 2.0):
                    with self.subTest(case=name, dtype=dtype, temperature=temperature):
                        targets = tensor(values)
                        loss_func = losses.RankNContrastLoss(temperature=temperature)
                        helpers.assert_matches_float64(
                            lambda x, f=loss_func, y=targets: f(x, y),
                            lambda x, y=values, t=temperature: equations_1_to_3(
                                x, y, t
                            ),
                            (batch(dtype),),
                            dtype,
                        )

    def test_hand_computed_value(self):
        embeddings = tensor([[0.0], [1.0], [3.0]])
        targets = tensor([0.0, 1.0, 5.0])
        loss = losses.RankNContrastLoss(temperature=1.0)(embeddings, targets)
        e = torch.exp(tensor(1.0))
        anchor_0 = (-torch.log(e**-1 / (e**-1 + e**-3)) - torch.log(tensor(1.0))) / 2
        anchor_1 = (-torch.log(e**-1 / (e**-1 + e**-2)) - torch.log(tensor(1.0))) / 2
        anchor_2 = (-torch.log(e**-2 / (e**-2 + e**-3)) - torch.log(tensor(1.0))) / 2
        expected = (anchor_0 + anchor_1 + anchor_2) / 3
        self.assertAlmostEqual(loss.item(), expected.item())

    def test_tied_samples_share_a_denominator(self):
        embeddings = tensor([[0.0], [1.0], [2.0]])
        targets = tensor([5.0, 4.0, 6.0])
        terms = losses.RankNContrastLoss(temperature=1.0).loss_terms(
            embeddings, targets
        )
        e = torch.exp(tensor(1.0))
        expected = -torch.log(e**-1 / (e**-1 + e**-2)) / 2
        expected = expected - torch.log(e**-2 / (e**-1 + e**-2)) / 2
        self.assertAlmostEqual(terms["loss"]["losses"][0].item(), expected.item())

    def test_ordered_embeddings_score_lower_than_shuffled(self):
        targets = tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        ordered = targets.unsqueeze(1)
        shuffled = ordered[torch.tensor([3, 0, 5, 1, 4, 2], device=TEST_DEVICE)]
        loss_func = losses.RankNContrastLoss(temperature=0.5)
        self.assertLess(
            loss_func(ordered, targets).item(), loss_func(shuffled, targets).item()
        )

    def test_row_order_does_not_matter(self):
        embeddings = batch()
        targets = tensor(CASES["duplicate-targets"])
        order = torch.randperm(7, device=TEST_DEVICE)
        loss_func = losses.RankNContrastLoss()
        self.assertAlmostEqual(
            loss_func(embeddings, targets).item(),
            loss_func(embeddings[order], targets[order]).item(),
        )

    def test_two_views(self):
        first = batch(size=4)
        second = first + 0.25
        targets = tensor([1.0, 5.0, 2.0, 9.0])
        embeddings = torch.cat([first, second])
        values = targets.repeat(2)
        loss = losses.RankNContrastLoss()(embeddings, values)
        self.assertAlmostEqual(
            loss.item(), equations_1_to_3(embeddings, values.tolist()).item()
        )

    def test_targets_are_not_rounded_to_the_embedding_dtype(self):
        targets = tensor([0.0, 1.0, 1.0 + 1e-9, 2.0])
        rounded = targets.to(torch.float32).double()
        self.assertEqual(rounded[1], rounded[2])
        embeddings = batch(torch.float32, size=4)
        loss_func = losses.RankNContrastLoss(temperature=0.5)
        wide = embeddings.double()
        expected = equations_1_to_3(wide, targets.tolist(), temperature=0.5)
        tied = equations_1_to_3(wide, rounded.tolist(), temperature=0.5)
        self.assertGreater(abs(expected.item() - tied.item()), 1e-3)
        self.assertAlmostEqual(
            loss_func(embeddings, targets).item(), expected.item(), places=5
        )

    def test_multidimensional_targets_use_summed_l1(self):
        targets = tensor([[0.0, 1.0], [2.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
        embeddings = batch(size=4)
        loss_func = losses.RankNContrastLoss(label_ndim=2)
        score = torch.linalg.vector_norm(
            embeddings[:, None] - embeddings[None, :], dim=-1
        )
        rows = range(4)
        gaps = [
            [(targets[i] - targets[j]).abs().sum().item() for j in rows] for i in rows
        ]
        anchors = []
        for i in rows:
            terms = []
            for j in rows:
                if i != j:
                    keep = [k for k in rows if k != i and gaps[i][k] >= gaps[i][j]]
                    denominator = torch.exp(-score[i, keep] / 2).sum()
                    terms.append(score[i, j] / 2 + torch.log(denominator))
            anchors.append(torch.stack(terms).mean())
        self.assertAlmostEqual(
            loss_func(embeddings, targets).item(), torch.stack(anchors).mean().item()
        )

    def test_custom_label_distance(self):
        def circular(x, y):
            gap = (x[:, None] - y[None, :]).abs() % 360
            return torch.minimum(gap, 360 - gap)

        targets = tensor([0.0, 350.0, 20.0, 180.0])
        embeddings = batch(size=4)
        loss = losses.RankNContrastLoss(label_distance=circular)(embeddings, targets)
        score = torch.linalg.vector_norm(
            embeddings[:, None] - embeddings[None, :], dim=-1
        )
        gaps = circular(targets, targets)
        rows = range(4)
        anchors = []
        for i in rows:
            terms = []
            for j in rows:
                if i != j:
                    keep = [k for k in rows if k != i and gaps[i, k] >= gaps[i, j]]
                    terms.append(
                        score[i, j] / 2
                        + torch.log(torch.exp(-score[i, keep] / 2).sum())
                    )
            anchors.append(torch.stack(terms).mean())
        self.assertAlmostEqual(loss.item(), torch.stack(anchors).mean().item())
        linear = losses.RankNContrastLoss()(embeddings, targets)
        self.assertNotAlmostEqual(loss.item(), linear.item(), places=3)

    def test_custom_label_distance_contract(self):
        embeddings = batch(size=3)
        targets = tensor([0.0, 1.0, 2.0])
        wrong = losses.RankNContrastLoss(label_distance=lambda x, y: x)
        with self.assertRaisesRegex(ValueError, "must return shape"):
            wrong(embeddings, targets)
        invalid = losses.RankNContrastLoss(
            label_distance=lambda x, y: torch.full((3, 3), torch.nan).to(x)
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            invalid(embeddings, targets)

    def test_separate_reference_set_excludes_nothing(self):
        embeddings = batch(size=3)
        reference = batch(size=4) + 0.5
        targets = tensor([1.0, 4.0, 2.0])
        ref_targets = tensor([1.0, 3.0, 5.0, 2.0])
        loss = losses.RankNContrastLoss()(
            embeddings, targets, ref_emb=reference, ref_labels=ref_targets
        )
        score = torch.linalg.vector_norm(
            embeddings[:, None] - reference[None, :], dim=-1
        )
        anchors = []
        for i in range(3):
            gaps = (targets[i] - ref_targets).abs()
            terms = []
            for j in range(4):
                keep = gaps >= gaps[j]
                terms.append(
                    score[i, j] / 2 + torch.log(torch.exp(-score[i, keep] / 2).sum())
                )
            anchors.append(torch.stack(terms).mean())
        self.assertAlmostEqual(loss.item(), torch.stack(anchors).mean().item())

    def test_similarity_distance(self):
        embeddings = batch()
        targets = tensor(CASES["distinct"])
        loss = losses.RankNContrastLoss(distance=distances.CosineSimilarity())(
            embeddings, targets
        )
        unit = torch.nn.functional.normalize(embeddings)
        expected = equations_1_to_3(
            embeddings, targets.tolist(), sign=1, score=unit @ unit.T
        )
        self.assertAlmostEqual(loss.item(), expected.item())

    def test_precomputed_matrix(self):
        embeddings = batch().requires_grad_()
        targets = tensor(CASES["duplicate-targets"])
        loss_func = losses.RankNContrastLoss()
        expected = loss_func(embeddings, targets)
        actual = loss_func(embeddings, targets, mat=loss_func.distance(embeddings))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_sum_reducer_scales_by_anchors(self):
        embeddings = batch()
        targets = tensor(CASES["distinct"])
        total = losses.RankNContrastLoss(reducer=reducers.SumReducer())(
            embeddings, targets
        )
        mean = losses.RankNContrastLoss()(embeddings, targets)
        self.assertAlmostEqual(total.item(), mean.item() * 7)

    def test_one_sample_gives_a_differentiable_zero(self):
        embeddings = batch(size=1).requires_grad_()
        loss = losses.RankNContrastLoss()(embeddings, tensor([1.0]))
        self.assertEqual(loss.item(), 0)
        loss.backward()

    def test_identical_embeddings_keep_finite_gradients(self):
        embeddings = torch.ones(4, 3, dtype=torch.float64, device=TEST_DEVICE)
        embeddings = embeddings.requires_grad_()
        loss = losses.RankNContrastLoss()(embeddings, tensor([1.0, 2.0, 3.0, 4.0]))
        self.assertTrue(loss.isfinite())
        loss.backward()
        assert embeddings.grad is not None
        self.assertTrue(embeddings.grad.isfinite().all())

    def test_target_contract(self):
        embeddings = batch(size=3)
        loss_func = losses.RankNContrastLoss()
        with self.assertRaisesRegex(TypeError, "floating-point"):
            loss_func(embeddings, torch.tensor([0, 1, 2], device=TEST_DEVICE))
        for bad in (torch.nan, torch.inf):
            with (
                self.subTest(bad=bad),
                self.assertRaisesRegex(ValueError, "finite"),
            ):
                loss_func(embeddings, tensor([0.0, bad, 2.0]))
        with self.assertRaises(ValueError):
            loss_func(embeddings)
        with self.assertRaises(ValueError):
            loss_func(embeddings, tensor([[0.0], [1.0], [2.0]]))
        with self.assertRaises(ValueError):
            pairs = tuple(torch.tensor([0], device=TEST_DEVICE) for _ in range(4))
            loss_func(embeddings, tensor([0.0, 1.0, 2.0]), pairs)
        with self.assertRaisesRegex(TypeError, "ref_labels"):
            loss_func(
                embeddings,
                tensor([0.0, 1.0, 2.0]),
                ref_emb=embeddings,
                ref_labels=torch.tensor([0, 1, 2], device=TEST_DEVICE),
            )

    def test_invalid_settings(self):
        for value in (0, -1):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "temperature"),
            ):
                losses.RankNContrastLoss(temperature=value)
        with self.assertRaisesRegex(ValueError, "label_ndim"):
            losses.RankNContrastLoss(label_ndim=3)
