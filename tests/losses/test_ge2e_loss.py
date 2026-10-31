import math
import unittest
from typing import Any, cast

import torch

from pytorch_metric_learning import distances, losses

from .. import TEST_DEVICE, TEST_DTYPES, helpers


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def equations_6_to_9(x, labels, w, b, variant):
    e = unit(x)
    groups = {}
    for row, label in enumerate(labels):
        groups.setdefault(label, []).append(row)
    terms = []
    for j, members in groups.items():
        if len(members) < 2:
            continue
        for i in members:
            own = e[[m for m in members if m != i]].mean(dim=0)
            positive = w * (e[i] @ unit(own)) + b
            others = [
                w * (e[i] @ unit(e[rows].mean(dim=0))) + b
                for k, rows in groups.items()
                if k != j
            ]
            if variant == "softmax":
                every = torch.stack([positive, *others])
                terms.append(-positive + torch.log(torch.exp(every).sum()))
            else:
                hardest = torch.stack([torch.sigmoid(s) for s in others]).max()
                terms.append(1 - torch.sigmoid(positive) + hardest)
    return torch.stack(terms).mean()


def rows_for(labels, dtype=torch.float64):
    generator = torch.Generator().manual_seed(3)
    return torch.randn(len(labels), 6, generator=generator, dtype=torch.float64).to(
        TEST_DEVICE, dtype
    )


def ge2e(variant="softmax", dtype=torch.float64, **options):
    return losses.GE2ELoss(variant=variant, **options).to(TEST_DEVICE, dtype)


CASES = {
    "balanced": [0, 0, 0, 1, 1, 1, 2, 2, 2],
    "uneven": [0, 0, 1, 1, 1, 1, 2, 2, 2],
    "unsorted": [2, 0, 1, 1, 0, 2, 1],
    "with-singleton": [0, 0, 0, 1, 1, 2],
}


class TestGE2ELoss(unittest.TestCase):
    def test_equations_on_embedding_gradients(self):
        for name, labels in CASES.items():
            for variant in ("softmax", "contrast"):
                for dtype in TEST_DTYPES:
                    with self.subTest(case=name, variant=variant, dtype=dtype):
                        y = torch.tensor(labels, device=TEST_DEVICE)
                        loss_func = ge2e(variant, dtype, init_w=2.0, init_b=-1.0)
                        helpers.assert_matches_float64(
                            lambda x, f=loss_func, y=y: f(x, y),
                            lambda x, y=labels, v=variant: equations_6_to_9(
                                x, y, 2.0, -1.0, v
                            ),
                            (rows_for(labels, dtype),),
                            dtype,
                        )

    def test_gradients_reach_w_and_b(self):
        labels = CASES["uneven"]
        y = torch.tensor(labels, device=TEST_DEVICE)
        x = rows_for(labels)
        for variant in ("softmax", "contrast"):
            with self.subTest(variant=variant):
                loss_func = ge2e(variant, init_w=2.0, init_b=-1.0)
                w = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
                b = torch.tensor(-1.0, dtype=torch.float64, requires_grad=True)
                reference = equations_6_to_9(x, labels, w, b, variant)
                expected_w, expected_b = torch.autograd.grad(reference, (w, b))
                loss_func(x, y).backward()
                assert loss_func.w.grad is not None and loss_func.b.grad is not None
                self.assertAlmostEqual(loss_func.w.grad.item(), expected_w.item())
                self.assertAlmostEqual(loss_func.b.grad.item(), expected_b.item())

    def test_b_cancels_in_softmax_only(self):
        labels = CASES["balanced"]
        y = torch.tensor(labels, device=TEST_DEVICE)
        x = rows_for(labels)
        first = ge2e("softmax", init_b=-5.0)(x, y)
        second = ge2e("softmax", init_b=3.0)(x, y)
        self.assertAlmostEqual(first.item(), second.item())
        self.assertNotAlmostEqual(
            ge2e("contrast", init_b=-5.0)(x, y).item(),
            ge2e("contrast", init_b=3.0)(x, y).item(),
        )

    def test_hand_computed_softmax(self):
        x = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=torch.float64,
            device=TEST_DEVICE,
        )
        y = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        loss = ge2e("softmax", init_w=2.0, init_b=0.0)(x, y)
        expected = -2 + math.log(math.exp(2) + 1)
        self.assertAlmostEqual(loss.item(), expected)

    def test_hand_computed_contrast(self):
        x = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=torch.float64,
            device=TEST_DEVICE,
        )
        y = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        loss = ge2e("contrast", init_w=2.0, init_b=-1.0)(x, y)
        self.assertAlmostEqual(
            loss.item(), 1 - 1 / (1 + math.exp(-1)) + 1 / (1 + math.exp(1))
        )

    def test_own_centroid_leaves_out_the_anchor(self):
        x = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [-1.0, 0.0]],
            dtype=torch.float64,
            device=TEST_DEVICE,
        )
        y = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        terms = ge2e("softmax", init_w=1.0, init_b=0.0).loss_terms(x, y)["loss"]
        expected = math.log(math.exp(0.0) + math.exp(-1.0)) - 0.0
        self.assertAlmostEqual(terms["losses"][0].item(), expected)

    def test_negative_centroids_keep_every_member(self):
        x = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=torch.float64,
            device=TEST_DEVICE,
        )
        y = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        terms = ge2e("softmax", init_w=1.0, init_b=0.0).loss_terms(x, y)["loss"]
        negative = unit(unit(x[2:]).mean(dim=0))
        expected = -1 + math.log(math.e + math.exp(negative[0].item()))
        self.assertAlmostEqual(terms["losses"][0].item(), expected)

    def test_row_scaling_changes_nothing(self):
        labels = CASES["uneven"]
        y = torch.tensor(labels, device=TEST_DEVICE)
        x = rows_for(labels)
        scale = torch.linspace(0.1, 50, len(labels), dtype=torch.float64).unsqueeze(1)
        for variant in ("softmax", "contrast"):
            with self.subTest(variant=variant):
                loss_func = ge2e(variant)
                self.assertAlmostEqual(
                    loss_func(x, y).item(),
                    loss_func(x * scale.to(TEST_DEVICE), y).item(),
                )

    def test_singleton_competes_as_a_negative(self):
        x = rows_for([0, 0, 0, 1])
        y = torch.tensor([0, 0, 0, 1], device=TEST_DEVICE)
        terms = ge2e().loss_terms(x, y)["loss"]
        assert isinstance(terms["indices"], torch.Tensor)
        self.assertEqual(terms["indices"].tolist(), [0, 1, 2])
        without = ge2e().loss_terms(x[:3], y[:3])
        self.assertEqual(without, {})

    def test_no_scored_row_gives_a_differentiable_zero(self):
        for labels in ([0, 1, 2], [0, 0, 0]):
            for variant in ("softmax", "contrast"):
                with self.subTest(labels=labels, variant=variant):
                    x = rows_for(labels).requires_grad_()
                    y = torch.tensor(labels, device=TEST_DEVICE)
                    loss = ge2e(variant)(x, y)
                    self.assertEqual(loss.item(), 0)
                    loss.backward()

    def test_w_stays_positive_without_changing_the_parameter(self):
        labels = CASES["balanced"]
        y = torch.tensor(labels, device=TEST_DEVICE)
        x = rows_for(labels)
        loss_func = ge2e("softmax")
        with torch.no_grad():
            loss_func.w.fill_(-3.0)
        loss = loss_func(x, y)
        self.assertAlmostEqual(loss.item(), math.log(3))
        self.assertEqual(loss_func.w.item(), -3.0)

    def test_parameters_are_saved_and_train(self):
        loss_func = ge2e()
        self.assertEqual(set(loss_func.state_dict()), {"w", "b"})
        self.assertEqual(len(list(loss_func.parameters())), 2)
        labels = CASES["balanced"]
        y = torch.tensor(labels, device=TEST_DEVICE)
        before = loss_func.w.item()
        optimizer = torch.optim.SGD(loss_func.parameters(), lr=1.0)
        loss_func(rows_for(labels), y).backward()
        optimizer.step()
        self.assertNotEqual(loss_func.w.item(), before)

    def test_invalid_arguments(self):
        with self.assertRaisesRegex(ValueError, "variant"):
            losses.GE2ELoss(variant=cast(Any, "other"))
        for value in (0.0, -1.0):
            with self.assertRaisesRegex(ValueError, "init_w"):
                losses.GE2ELoss(init_w=value)
        with self.assertRaisesRegex(ValueError, "requires a similarity"):
            losses.GE2ELoss(distance=distances.LpDistance())

    def test_rejects_unsupported_inputs(self):
        x = rows_for([0, 0, 1, 1])
        y = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        loss_func = ge2e()
        pairs = tuple(torch.tensor([0], device=TEST_DEVICE) for _ in range(4))
        with self.assertRaises(ValueError):
            loss_func(x, y, pairs)
        with self.assertRaises(ValueError):
            loss_func(x, y, ref_emb=x, ref_labels=y)
        with self.assertRaises(ValueError):
            loss_func(x)
        with self.assertRaisesRegex(ValueError, "Move the loss"):
            losses.GE2ELoss().to(TEST_DEVICE)(x, y)
