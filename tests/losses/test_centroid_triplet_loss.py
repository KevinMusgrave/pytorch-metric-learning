import unittest

import torch

from pytorch_metric_learning import distances, losses, reducers

from .. import TEST_DEVICE, TEST_DTYPES, helpers

REPORTED_CLASSES = (
    "228 83 200 254 250 13 124 86 179 338 146 225 22 73 300 431 349 305 117 56 206 "
    "293 119 135 394 126 366 59 235 352 193 341"
)
REPORTED_RUNS = [
    (int(label), 2 if label == "56" else 10) for label in REPORTED_CLASSES.split()
]


def repeated(runs):
    return [label for label, count in runs for _ in range(count)]


def equation_2(embeddings, labels, margin=0.05, smooth=False):
    groups = {}
    for row, label in enumerate(labels):
        groups.setdefault(label, []).append(row)
    terms = []
    for label, members in groups.items():
        if len(members) < 2:
            continue
        for anchor in members:
            others = [m for m in members if m != anchor]
            positive = embeddings[others].mean(dim=0)
            positive_distance = (embeddings[anchor] - positive).pow(2).sum()
            for other_label, other_members in groups.items():
                if other_label == label:
                    continue
                negative = embeddings[other_members].mean(dim=0)
                negative_distance = (embeddings[anchor] - negative).pow(2).sum()
                terms.append(positive_distance - negative_distance + margin)
    values = torch.stack(terms)
    if smooth:
        return torch.nn.functional.softplus(values).mean()
    values = torch.relu(values)
    return values.sum() / (values > 0).sum().clamp_min(1)


def rows_for(labels, dtype=torch.float64, dimensions=8):
    generator = torch.Generator().manual_seed(9)
    return torch.randn(
        len(labels), dimensions, generator=generator, dtype=torch.float64
    ).to(TEST_DEVICE, dtype)


class TestCentroidTripletLoss(unittest.TestCase):
    def test_equation_2_on_balanced_and_uneven_classes(self):
        cases = {
            "balanced": [0, 0, 0, 1, 1, 1],
            "uneven": [0, 0, 1, 1, 1, 2, 2],
            "unsorted": [2, 0, 1, 1, 0, 2, 1],
            "with-singleton": [0, 0, 1, 1, 1, 2, 2, 3],
        }
        for name, labels in cases.items():
            for dtype in TEST_DTYPES:
                for smooth in (False, True):
                    with self.subTest(case=name, dtype=dtype, smooth=smooth):
                        y = torch.tensor(labels, device=TEST_DEVICE)
                        options = {"reducer": reducers.MeanReducer()} if smooth else {}
                        loss_func = losses.CentroidTripletLoss(
                            smooth_loss=smooth, **options
                        )
                        helpers.assert_matches_float64(
                            lambda x, f=loss_func, y=y: f(x, y),
                            lambda x, y=labels, s=smooth: equation_2(x, y, smooth=s),
                            (rows_for(labels, dtype),),
                            dtype,
                        )

    def test_reported_labels_with_one_small_class(self):
        labels = repeated(REPORTED_RUNS)
        self.assertEqual(len(labels), 312)
        embeddings = rows_for(labels, dimensions=10).requires_grad_()
        y = torch.tensor(labels, device=TEST_DEVICE)
        loss = losses.CentroidTripletLoss()(embeddings, y)
        expected = equation_2(embeddings, labels)
        self.assertAlmostEqual(loss.item(), expected.item())
        gradient = torch.autograd.grad(loss, embeddings, retain_graph=True)[0]
        reference = torch.autograd.grad(expected, embeddings)[0]
        torch.testing.assert_close(gradient, reference)

    def test_reported_labels_with_two_small_classes(self):
        runs = list(REPORTED_RUNS)
        runs.insert(21, (1, 2))
        labels = repeated(runs)
        self.assertEqual(len(labels), 314)
        embeddings = rows_for(labels, dimensions=10)
        y = torch.tensor(labels, device=TEST_DEVICE)
        self.assertAlmostEqual(
            losses.CentroidTripletLoss()(embeddings, y).item(),
            equation_2(embeddings, labels).item(),
        )

    def test_original_failure_examples_now_compute(self):
        for labels in ([0] * 8 + [1], [0, 0, 1, 1, 1, 2, 2, 3]):
            with self.subTest(labels=labels):
                embeddings = rows_for(labels)
                y = torch.tensor(labels, device=TEST_DEVICE)
                self.assertAlmostEqual(
                    losses.CentroidTripletLoss()(embeddings, y).item(),
                    equation_2(embeddings, labels).item(),
                )

    def test_singleton_is_a_negative_but_not_an_anchor(self):
        embeddings = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]],
            dtype=torch.float64,
            device=TEST_DEVICE,
        )
        labels = torch.tensor([0, 0, 1], device=TEST_DEVICE)
        loss_func = losses.CentroidTripletLoss(
            margin=0, reducer=reducers.MeanReducer(), smooth_loss=True
        )
        terms = loss_func.loss_terms(embeddings, labels)["loss"]
        assert isinstance(terms["indices"], torch.Tensor)
        self.assertEqual(terms["indices"].tolist(), [0, 1])
        expected = torch.nn.functional.softplus(
            torch.tensor([1.0 - 9.0, 1.0 - 4.0], dtype=torch.float64)
        ).to(TEST_DEVICE)
        torch.testing.assert_close(terms["losses"], expected)

    def test_leave_one_out_differs_from_the_full_centroid(self):
        embeddings = torch.tensor(
            [[0.0, 0.0], [4.0, 0.0], [10.0, 0.0], [10.0, 0.0]],
            dtype=torch.float64,
            device=TEST_DEVICE,
        )
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        loss = losses.CentroidTripletLoss(margin=200)(embeddings, labels)
        anchor_0 = 16 - 100 + 200
        anchor_1 = 16 - 36 + 200
        anchor_2 = 0 - 64 + 200
        self.assertAlmostEqual(loss.item(), (anchor_0 + anchor_1 + 2 * anchor_2) / 4)

    def test_row_order_does_not_matter(self):
        labels = [0, 0, 1, 1, 1, 2, 2, 3]
        embeddings = rows_for(labels)
        y = torch.tensor(labels, device=TEST_DEVICE)
        order = torch.randperm(len(labels), device=TEST_DEVICE)
        loss_func = losses.CentroidTripletLoss()
        self.assertAlmostEqual(
            loss_func(embeddings, y).item(),
            loss_func(embeddings[order], y[order]).item(),
        )

    def test_sparse_labels_match_contiguous_labels(self):
        embeddings = rows_for([0] * 6)
        first = torch.tensor([0, 0, 1, 1, 2, 2], device=TEST_DEVICE)
        second = torch.tensor([2**40, 2**40, 7, 7, 300, 300], device=TEST_DEVICE)
        loss_func = losses.CentroidTripletLoss()
        self.assertAlmostEqual(
            loss_func(embeddings, first).item(), loss_func(embeddings, second).item()
        )

    def test_other_distances(self):
        labels = [0, 0, 1, 1, 1]
        y = torch.tensor(labels, device=TEST_DEVICE)
        embeddings = rows_for(labels)
        unit = torch.nn.functional.normalize
        cosine = losses.CentroidTripletLoss(
            distance=distances.CosineSimilarity(), margin=0.3, smooth_loss=True
        )
        groups = {0: [0, 1], 1: [2, 3, 4]}
        terms = []
        for label, members in groups.items():
            negative = unit(embeddings[groups[1 - label]].mean(dim=0), dim=0)
            for anchor in members:
                others = [m for m in members if m != anchor]
                positive = unit(embeddings[others].mean(dim=0), dim=0)
                query = unit(embeddings[anchor], dim=0)
                terms.append(query @ negative - query @ positive + 0.3)
        expected = torch.nn.functional.softplus(torch.stack(terms)).mean()
        self.assertAlmostEqual(cosine(embeddings, y).item(), expected.item())

    def test_no_anchor_or_one_class_gives_a_differentiable_zero(self):
        for labels in ([0, 1, 2, 3], [0, 0, 0, 0]):
            with self.subTest(labels=labels):
                embeddings = rows_for(labels).requires_grad_()
                y = torch.tensor(labels, device=TEST_DEVICE)
                loss = losses.CentroidTripletLoss()(embeddings, y)
                self.assertEqual(loss.item(), 0)
                loss.backward()
                assert embeddings.grad is not None
                self.assertEqual(embeddings.grad.abs().sum().item(), 0)

    def test_low_precision_class_sum_does_not_overflow(self):
        size = 70000
        embeddings = torch.ones(size, 1, dtype=torch.float16, device=TEST_DEVICE)
        embeddings[size // 2 :] = 3.0
        y = (torch.arange(size) >= size // 2).long().to(TEST_DEVICE)
        loss = losses.CentroidTripletLoss(margin=9)(embeddings, y)
        self.assertTrue(loss.isfinite())
        self.assertAlmostEqual(loss.item(), 9 - 4, places=1)

    def test_rejects_unsupported_inputs(self):
        embeddings = rows_for([0, 0, 1, 1])
        y = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        loss_func = losses.CentroidTripletLoss()
        pairs = tuple(torch.tensor([0], device=TEST_DEVICE) for _ in range(4))
        with self.assertRaises(ValueError):
            loss_func(embeddings, y, pairs)
        with self.assertRaises(ValueError):
            loss_func(embeddings, y, ref_emb=embeddings, ref_labels=y)
        with self.assertRaises(ValueError):
            loss_func(embeddings)
