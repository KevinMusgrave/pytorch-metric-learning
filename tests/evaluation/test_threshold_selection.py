import unittest

import torch

from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.evaluation import (
    MatchCounts,
    MatchFinder,
    match_accuracy,
    match_balanced_accuracy,
    match_f1,
    select_threshold,
)

from .. import TEST_DEVICE, TEST_DTYPES


def counts_at(scores, labels, threshold, is_inverted):
    matched = scores >= threshold if is_inverted else scores <= threshold
    return MatchCounts(
        (matched & labels).sum().double().reshape(1),
        (matched & ~labels).sum().double().reshape(1),
        (~matched & ~labels).sum().double().reshape(1),
        (~matched & labels).sum().double().reshape(1),
    )


def brute_force_best(scores, labels, metric, is_inverted):
    values = torch.unique(scores).tolist()
    candidates = [*values, torch.inf if is_inverted else -torch.inf]
    results = [
        metric(counts_at(scores, labels, threshold, is_inverted)).item()
        for threshold in candidates
    ]
    return max(value for value in results if value == value)


class TestSelectThreshold(unittest.TestCase):
    def test_metrics_have_their_definitions(self):
        counts = MatchCounts(*torch.tensor([[6.0], [2.0], [9.0], [3.0]]))
        self.assertAlmostEqual(match_accuracy(counts).item(), 15 / 20)
        self.assertAlmostEqual(
            match_balanced_accuracy(counts).item(), (6 / 9 + 9 / 11) / 2
        )
        self.assertAlmostEqual(match_f1(counts).item(), 12 / 17)

    def test_matches_exhaustive_search_for_both_directions(self):
        generator = torch.Generator().manual_seed(4)
        metrics = (match_accuracy, match_balanced_accuracy, match_f1)
        for trial in range(30):
            rows = 40
            labels = torch.rand(rows, generator=generator) < 0.3
            labels[:2] = torch.tensor([True, False])
            noise = torch.randint(0, 12, (rows,), generator=generator)
            scores = (labels.double() * 3 + noise).to(torch.float64)
            for metric in metrics:
                for is_inverted in (True, False):
                    with self.subTest(trial=trial, metric=metric.__name__):
                        signed = scores if is_inverted else -scores
                        selection = select_threshold(
                            signed.to(TEST_DEVICE),
                            labels.to(TEST_DEVICE),
                            metric,
                            is_inverted=is_inverted,
                        )
                        self.assertAlmostEqual(
                            selection.value,
                            brute_force_best(signed, labels, metric, is_inverted),
                        )
                        counts = counts_at(
                            signed, labels, selection.threshold, is_inverted
                        )
                        self.assertAlmostEqual(metric(counts).item(), selection.value)

    def test_threshold_is_midpoint_of_separating_gap(self):
        distances = torch.tensor([0.1, 0.25, 0.75, 0.9], device=TEST_DEVICE)
        labels = torch.tensor([True, True, False, False], device=TEST_DEVICE)
        selection = select_threshold(distances, labels, match_f1, is_inverted=False)
        self.assertEqual(selection, (0.5, 1.0))
        similarities = -distances
        selection = select_threshold(similarities, labels, match_f1, is_inverted=True)
        self.assertEqual(selection, (-0.5, 1.0))

    def test_tied_scores_are_decided_together(self):
        scores = torch.tensor([0.1, 0.5, 0.5, 0.9], device=TEST_DEVICE)
        labels = torch.tensor([True, True, False, False], device=TEST_DEVICE)
        selection = select_threshold(scores, labels, match_accuracy, is_inverted=False)
        self.assertAlmostEqual(selection.threshold, 0.3)
        self.assertEqual(selection.value, 0.75)

    def test_metric_ties_choose_fewest_matches(self):
        scores = torch.tensor([3.0, 2.0, 1.0, 0.0], device=TEST_DEVICE)
        labels = torch.tensor([True, False, True, False], device=TEST_DEVICE)
        selection = select_threshold(scores, labels, match_accuracy, is_inverted=True)
        self.assertEqual(selection, (2.5, 0.75))

    def test_extreme_candidates(self):
        scores = torch.tensor([0.0, 1.0, 2.0, 3.0], device=TEST_DEVICE)
        labels = torch.tensor([False, False, False, True], device=TEST_DEVICE)
        for is_inverted, expected in ((False, -torch.inf), (True, 2.5)):
            selection = select_threshold(
                scores, labels, match_accuracy, is_inverted=is_inverted
            )
            self.assertEqual(selection.threshold, expected)
        everything = select_threshold(
            scores,
            labels,
            lambda counts: counts.true_positives + counts.false_positives,
            is_inverted=False,
        )
        self.assertEqual(everything, (3.0, 4.0))

    def test_nothing_matches_with_inverted_scores(self):
        scores = torch.tensor([3.0, 2.0, 1.0, 0.0], device=TEST_DEVICE)
        labels = torch.tensor([False, False, False, True], device=TEST_DEVICE)
        selection = select_threshold(scores, labels, match_accuracy, is_inverted=True)
        self.assertEqual(selection, (torch.inf, 0.75))
        finder = MatchFinder(CosineSimilarity(), threshold=selection.threshold)
        embeddings = torch.eye(2, device=TEST_DEVICE)
        self.assertFalse(finder.get_match_matrix(embeddings).any())

    def test_threshold_reproduces_value_under_match_finder(self):
        for dtype in TEST_DTYPES:
            for distance in (CosineSimilarity(), LpDistance()):
                with self.subTest(dtype=dtype, distance=type(distance).__name__):
                    generator = torch.Generator().manual_seed(9)
                    first = torch.randn(60, 8, generator=generator, dtype=torch.float64)
                    second = torch.randn(
                        60, 8, generator=generator, dtype=torch.float64
                    )
                    labels = torch.arange(60) % 3 == 0
                    second[labels] = first[labels] + 0.8 * second[labels]
                    first = first.to(TEST_DEVICE, dtype)
                    second = second.to(TEST_DEVICE, dtype)
                    selection = select_threshold(
                        distance.pairwise(first, second),
                        labels,
                        match_f1,
                        is_inverted=distance.is_inverted,
                    )
                    finder = MatchFinder(distance, threshold=selection.threshold)
                    matched = finder.is_match(first, second).cpu()
                    recovered = MatchCounts(
                        (matched & labels).sum().double(),
                        (matched & ~labels).sum().double(),
                        (~matched & ~labels).sum().double(),
                        (~matched & labels).sum().double(),
                    )
                    self.assertAlmostEqual(match_f1(recovered).item(), selection.value)

    def test_adjacent_scores_keep_the_decision(self):
        one = torch.tensor(1.0, dtype=torch.float32)
        two = torch.tensor(2.0, dtype=torch.float32)
        odd = torch.nextafter(one, two)
        for kept, excluded in ((one, odd), (odd, torch.nextafter(odd, two))):
            scores = torch.stack([kept, excluded])
            labels = torch.tensor([True, False])
            selection = select_threshold(scores, labels, match_f1, is_inverted=False)
            self.assertEqual(selection.value, 1.0)
            self.assertEqual((scores <= selection.threshold).tolist(), labels.tolist())

    def test_undefined_values_are_never_selected(self):
        scores = torch.tensor([0.0, 1.0, 2.0], device=TEST_DEVICE)
        labels = torch.tensor([True, False, False], device=TEST_DEVICE)

        def defined_for_two(counts):
            values = counts.true_positives + counts.false_positives
            return torch.where(values == 2, values, torch.nan)

        selection = select_threshold(scores, labels, defined_for_two, is_inverted=False)
        self.assertEqual(selection, (1.5, 2.0))

    def test_inputs_are_not_tracked_or_changed(self):
        scores = torch.tensor([0.5, 0.1, 0.9], requires_grad=True, device=TEST_DEVICE)
        labels = torch.tensor([True, True, False], device=TEST_DEVICE)
        before = scores.detach().clone()
        select_threshold(scores, labels, match_f1, is_inverted=False)
        self.assertTrue(torch.equal(scores.detach(), before))
        self.assertIsNone(scores.grad)

    def test_numpy_inputs(self):
        selection = select_threshold(
            torch.tensor([0.1, 0.9]).numpy(),
            torch.tensor([True, False]).numpy(),
            match_f1,
            is_inverted=False,
        )
        self.assertAlmostEqual(selection.threshold, 0.5)

    def test_invalid_inputs(self):
        scores = torch.tensor([0.0, 1.0])
        labels = torch.tensor([True, False])
        cases = [
            (scores.unsqueeze(0), labels, "one-dimensional"),
            (scores, labels[:1], "one-dimensional"),
            (scores.long(), labels, "floating point"),
            (scores, labels.long(), "boolean"),
            (torch.tensor([0.0, torch.nan]), labels, "finite"),
            (scores, torch.tensor([True, True]), "matching and non-matching"),
            (scores, torch.tensor([False, False]), "matching and non-matching"),
        ]
        for bad_scores, bad_labels, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                select_threshold(bad_scores, bad_labels, match_f1, is_inverted=False)

    def test_invalid_metric_output(self):
        scores = torch.tensor([0.0, 1.0])
        labels = torch.tensor([True, False])
        with self.assertRaisesRegex(ValueError, "one value per candidate"):
            select_threshold(
                scores, labels, lambda c: c.true_positives[:1], is_inverted=False
            )
        with self.assertRaisesRegex(ValueError, "undefined for every"):
            select_threshold(
                scores,
                labels,
                lambda c: c.true_positives * torch.nan,
                is_inverted=False,
            )
