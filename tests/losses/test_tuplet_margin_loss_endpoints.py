import copy
import math
import unittest

import torch

from pytorch_metric_learning.losses import TupletMarginLoss

from .. import TEST_DEVICE, TEST_DTYPES


def scalar_tuplet(query, positive, negative, margin, scale):
    def cosine(x, y):
        return sum(a * b for a, b in zip(x, y)) / math.sqrt(
            sum(a * a for a in x) * sum(b * b for b in y)
        )

    positive_score = cosine(query, positive)
    angle = math.acos(max(-1.0, min(1.0, positive_score)))
    difference = scale * (
        cosine(query, negative) - math.cos(angle - math.radians(margin))
    )
    return max(difference, 0) + math.log1p(math.exp(-abs(difference)))


class TestTupletMarginLossEndpoints(unittest.TestCase):
    def inputs(self, dtype, sign, negative=(0.0, 1.0)):
        query = torch.tensor(
            [[1.0, 0.0]], device=TEST_DEVICE, dtype=dtype, requires_grad=True
        )
        reference = torch.tensor(
            [[sign, 0.0], negative],
            device=TEST_DEVICE,
            dtype=dtype,
            requires_grad=True,
        )
        return query, reference

    def evaluate(self, loss, query, reference, mode):
        if mode == "labels":
            return loss(
                query,
                torch.tensor([7], device=TEST_DEVICE),
                ref_emb=reference,
                ref_labels=torch.tensor([7, 9], device=TEST_DEVICE),
            )
        return loss(
            query,
            indices_tuple=tuple(
                torch.tensor(x, device=TEST_DEVICE) for x in ([0], [0], [1])
            ),
            ref_emb=reference,
        )

    def test_zero_margin_endpoint_values_and_gradients(self):
        # The zero-margin objective is two-logit cross entropy, with no angle.
        for dtype in TEST_DTYPES:
            for sign in (1.0, -1.0):
                for negative in ((0.0, 1.0), (1.0, 0.0), (-1.0, 0.0)):
                    for mode in ("labels", "triplet"):
                        with self.subTest(
                            dtype=dtype, sign=sign, negative=negative, mode=mode
                        ):
                            query, reference = self.inputs(dtype, sign, negative)
                            value = self.evaluate(
                                TupletMarginLoss(margin=0, scale=2),
                                query,
                                reference,
                                mode,
                            )
                            expected = scalar_tuplet(
                                [1.0, 0.0], [sign, 0.0], negative, 0, 2
                            )
                            self.assertAlmostEqual(value.item(), expected, places=6)
                            slope = 2 / (1 + math.exp(-2 * (negative[0] - sign)))
                            targets = (
                                [[0.0, slope * negative[1]]],
                                [
                                    [0.0, 0.0],
                                    [
                                        slope * (1 - negative[0] ** 2),
                                        -slope * negative[0] * negative[1],
                                    ],
                                ],
                            )
                            for grad, target in zip(
                                torch.autograd.grad(value, (query, reference)), targets
                            ):
                                self.assertTrue(
                                    torch.allclose(
                                        grad,
                                        grad.new_tensor(target),
                                        rtol=1e-5,
                                        atol=1e-7,
                                    ),
                                    repr(grad),
                                )

    def test_endpoint_optimizer_step_and_margin_variants(self):
        for dtype in TEST_DTYPES:
            for sign in (1.0, -1.0):
                for mode in ("labels", "triplet"):
                    for margin, scale in (
                        (0, 2),
                        (5.73, 64),
                        (-5.73, 2),
                        (45, 2),
                        (90, 2),
                        (180, 2),
                    ):
                        with self.subTest(
                            dtype=dtype, sign=sign, mode=mode, margin=margin
                        ):
                            query, reference = self.inputs(dtype, sign)
                            loss = TupletMarginLoss(
                                margin=margin, scale=scale, collect_stats=True
                            )
                            optimizer = torch.optim.SGD([query, reference], lr=0.01)
                            value = self.evaluate(loss, query, reference, mode)
                            expected = scalar_tuplet(
                                [1.0, 0.0], [sign, 0.0], [0.0, 1.0], margin, scale
                            )
                            self.assertTrue(
                                math.isclose(
                                    value.item(), expected, rel_tol=1e-5, abs_tol=1e-6
                                )
                            )
                            self.assertAlmostEqual(
                                loss.avg_pos_angle, 0 if sign > 0 else 180, places=4
                            )
                            self.assertAlmostEqual(loss.avg_neg_angle, 90, places=4)
                            value.backward()
                            for name, tensor in (
                                ("query", query),
                                ("reference", reference),
                            ):
                                with self.subTest(gradient=name):
                                    self.assertTrue(torch.isfinite(tensor.grad).all())
                            angle = 0 if sign > 0 else math.pi
                            adjusted_positive = math.cos(angle - math.radians(margin))
                            slope = scale / (1 + math.exp(scale * adjusted_positive))
                            with self.subTest(gradient="endpoint_policy"):
                                # The angular positive term has zero endpoint
                                # gradient; the negative term must remain active.
                                self.assertTrue(
                                    torch.allclose(
                                        query.grad,
                                        query.new_tensor([[0, slope]]),
                                        rtol=1e-4,
                                        atol=0,
                                    )
                                )
                                self.assertTrue(
                                    torch.allclose(
                                        reference.grad,
                                        reference.new_tensor([[0, 0], [slope, 0]]),
                                        rtol=1e-4,
                                        atol=0,
                                    )
                                )
                            # Step even if the subtest above failed: NaN gradients
                            # must also be shown to contaminate real parameters.
                            optimizer.step()
                            with self.subTest(optimizer="parameters"):
                                self.assertTrue(torch.isfinite(query).all())
                                self.assertTrue(torch.isfinite(reference).all())
                            if margin == 0:
                                slope = scale / (1 + math.exp(scale * sign))
                                with self.subTest(optimizer="independent_update"):
                                    self.assertTrue(
                                        torch.allclose(
                                            query,
                                            query.new_tensor([[1, -0.01 * slope]]),
                                        )
                                    )
                                    self.assertTrue(
                                        torch.allclose(
                                            reference,
                                            reference.new_tensor(
                                                [[sign, 0], [-0.01 * slope, 1]]
                                            ),
                                        )
                                    )

    def test_nonendpoint_values_and_gradients(self):
        # Coordinate finite differences of an independent Python scalar objective.
        original = [[1.0, 2.0, -1.0], [0.25, 1.0, -2.0], [-1.0, 0.5, 2.0]]
        for dtype in TEST_DTYPES:
            for margin in (-20, 0, 5.73, 30, 90, 180):
                with self.subTest(dtype=dtype, margin=margin):
                    query = torch.tensor(
                        [original[0]],
                        device=TEST_DEVICE,
                        dtype=dtype,
                        requires_grad=True,
                    )
                    reference = torch.tensor(
                        original[1:],
                        device=TEST_DEVICE,
                        dtype=dtype,
                        requires_grad=True,
                    )
                    value = self.evaluate(
                        TupletMarginLoss(margin=margin, scale=2),
                        query,
                        reference,
                        "triplet",
                    )
                    expected = scalar_tuplet(*original, margin, 2)
                    self.assertTrue(
                        math.isclose(value.item(), expected, rel_tol=1e-5, abs_tol=1e-6)
                    )
                    targets = []
                    for i, row in enumerate(original):
                        target = []
                        for j in range(len(row)):
                            plus, minus = (
                                copy.deepcopy(original),
                                copy.deepcopy(original),
                            )
                            plus[i][j] += 1e-5
                            minus[i][j] -= 1e-5
                            target.append(
                                (
                                    scalar_tuplet(*plus, margin, 2)
                                    - scalar_tuplet(*minus, margin, 2)
                                )
                                / 2e-5
                            )
                        targets.append(target)
                    for grad, target in zip(
                        torch.autograd.grad(value, (query, reference)),
                        ([targets[0]], targets[1:]),
                    ):
                        self.assertTrue(
                            torch.allclose(
                                grad, grad.new_tensor(target), rtol=2e-4, atol=2e-5
                            ),
                            repr(grad),
                        )

    def test_mixed_endpoint_and_regular_pairs_zero_margin(self):
        for dtype in TEST_DTYPES:
            for sign in (1.0, -1.0):
                with self.subTest(dtype=dtype, sign=sign):
                    query = torch.tensor(
                        [[1.0, 0.0], [1.0, 2.0]],
                        device=TEST_DEVICE,
                        dtype=dtype,
                        requires_grad=True,
                    )
                    reference = torch.tensor(
                        [[sign, 0.0], [0.0, 1.0], [2.0, -1.0], [-1.0, 1.0]],
                        device=TEST_DEVICE,
                        dtype=dtype,
                        requires_grad=True,
                    )
                    ids = tuple(
                        torch.tensor(x, device=TEST_DEVICE)
                        for x in ([0, 1], [0, 2], [0, 1], [1, 3])
                    )
                    normalized_q = torch.nn.functional.normalize(query)
                    normalized_r = torch.nn.functional.normalize(reference)
                    logits = 2 * (
                        normalized_q[:, None, :] * normalized_r.reshape(2, 2, 2)
                    ).sum(dim=2)
                    expected = torch.nn.functional.cross_entropy(
                        logits, torch.zeros(2, dtype=torch.long, device=TEST_DEVICE)
                    )
                    expected_gradients = torch.autograd.grad(
                        expected, (query, reference)
                    )
                    actual = TupletMarginLoss(margin=0, scale=2)(
                        query, indices_tuple=ids, ref_emb=reference
                    )
                    self.assertTrue(torch.allclose(actual, expected))
                    for grad, target in zip(
                        torch.autograd.grad(actual, (query, reference)),
                        expected_gradients,
                    ):
                        self.assertTrue(
                            torch.allclose(grad, target, rtol=1e-5, atol=1e-6),
                            repr(grad),
                        )
