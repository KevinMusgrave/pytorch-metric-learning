import math
import unittest

import torch

from pytorch_metric_learning.distances import DotProductSimilarity, LpDistance
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.reducers import DoNothingReducer

from .. import TEST_DEVICE


def scalar_pair_oracle(queries, references, positives, negatives, temperature):
    """Independent scalar probabilities and analytic first derivatives."""
    query_grad = [[0.0] * len(row) for row in queries]
    reference_grad = [[0.0] * len(row) for row in references]
    losses = []
    temperature_grad = 0.0
    for anchor, positive in positives:
        query = queries[anchor]
        positive_score = sum(x * y for x, y in zip(query, references[positive]))
        matching = [negative for owner, negative in negatives if owner == anchor]
        differences = [
            (sum(x * y for x, y in zip(query, references[negative])) - positive_score)
            / temperature
            for negative in matching
        ]
        pivot = max([0.0] + differences)
        loss = pivot + math.log(sum(math.exp(x - pivot) for x in [0.0] + differences))
        losses.append(loss)
        for negative, difference in zip(matching, differences):
            probability = math.exp(difference - loss)
            temperature_grad -= probability * difference / temperature
            for feature, coordinate in enumerate(query):
                query_grad[anchor][feature] += (
                    probability
                    * (references[negative][feature] - references[positive][feature])
                    / temperature
                )
                reference_grad[positive][feature] -= (
                    probability * coordinate / temperature
                )
                reference_grad[negative][feature] += (
                    probability * coordinate / temperature
                )
    return losses, query_grad, reference_grad, temperature_grad


def repeated_positive_pair_indices():
    # A real public pair tuple with two positive terms and one negative.
    # GenericPairLoss retains explicit four-tuples and needs at least one length>1.
    return tuple(
        torch.tensor(values, device=TEST_DEVICE)
        for values in ([0, 0], [0, 0], [0], [1])
    )


def mean_pair_oracle(queries, references, positives, negatives, temperature):
    losses, query_gradient, reference_gradient, temperature_gradient = (
        scalar_pair_oracle(queries, references, positives, negatives, temperature)
    )
    count = len(losses)
    return (
        [sum(losses) / count],
        [[value / count for value in row] for row in query_gradient],
        [[value / count for value in row] for row in reference_gradient],
        temperature_gradient / count,
    )


class OffsetDotProductSimilarity(DotProductSimilarity):
    """A legal public distance extension: add a constant to every similarity."""

    def compute_mat(self, query_emb, ref_emb):
        return super().compute_mat(query_emb, ref_emb) + 100000000.0

    def pairwise_distance(self, query_emb, ref_emb):
        return super().pairwise_distance(query_emb, ref_emb) + 100000000.0


class TestNTXentLogDomain(unittest.TestCase):
    def assert_close_tensor(self, actual, expected, rtol, atol):
        # torch.testing.assert_close is absent in the supported torch1.6 CI job.
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        self.assertTrue(
            torch.allclose(actual, expected, rtol=rtol, atol=atol),
            "actual={} expected={} rtol={} atol={}".format(
                actual, expected, rtol, atol
            ),
        )

    def require_half_backend(self, dtype):
        if dtype != torch.float16 or TEST_DEVICE.type != "cpu":
            return
        # Probe the real selected backend, not its version or the loss under test.
        # Only exact, explicit unsupported-Half errors may skip these new controls.
        operations = (
            (
                "CPU half matrix multiply forward/backward",
                lambda value: torch.mm(value, value),
                {
                    "\"addmm\" not implemented for 'Half'",
                    "\"addmm_impl_cpu_\" not implemented for 'Half'",
                },
            ),
            (
                "CPU half softplus forward/backward",
                torch.nn.functional.softplus,
                {
                    "\"softplus_cpu\" not implemented for 'Half'",
                    "\"softplus_backward_cpu\" not implemented for 'Half'",
                    "\"softplus_backward_cpu_out\" not implemented for 'Half'",
                },
            ),
        )
        for operation, compute, declared_unsupported in operations:
            value = torch.ones(
                (1, 1), dtype=torch.float16, device=TEST_DEVICE, requires_grad=True
            )
            try:
                result = compute(value)
                result.sum().backward()
            except RuntimeError as error:
                if str(error) in declared_unsupported:
                    self.skipTest("{}: {}".format(operation, error))
                raise
            self.assertTrue(torch.isfinite(result).all())
            self.assertTrue(torch.isfinite(value.grad).all())

    def assert_value_and_gradients(
        self, obtained, queries, references, expected, dtype
    ):
        """Check values and each input gradient independently of producer code."""
        losses, query_gradient, reference_gradient, _ = expected
        tolerance = (
            3e-3
            if dtype == torch.float16
            else 3e-5 if dtype == torch.float32 else 2e-12
        )
        obtained.sum().backward()
        for name, tensor in (
            ("loss", obtained),
            ("query gradient", queries.grad),
            ("reference gradient", references.grad),
        ):
            with self.subTest(finite=name):
                self.assertTrue(torch.isfinite(tensor).all())
        with self.subTest(oracle="loss"):
            self.assert_close_tensor(
                obtained.double(),
                torch.tensor(losses, device=TEST_DEVICE, dtype=torch.float64),
                rtol=tolerance,
                atol=tolerance,
            )
        for name, actual, reference in (
            ("query", queries.grad, query_gradient),
            ("reference", references.grad, reference_gradient),
        ):
            with self.subTest(oracle=name):
                self.assert_close_tensor(
                    actual.double(),
                    torch.tensor(reference, device=TEST_DEVICE, dtype=torch.float64),
                    rtol=tolerance,
                    atol=tolerance,
                )

    def repeated_pair_raw_losses(self, loss_dict):
        # A zero_losses early return has neither the two pair losses nor these indices.
        losses = loss_dict["loss"]["losses"]
        self.assertEqual(losses.shape, torch.Size([2]))
        for index in loss_dict["loss"]["indices"]:
            self.assertTrue(
                torch.equal(index, torch.tensor([0, 0], device=TEST_DEVICE))
            )
        return losses

    def single_pair(
        self,
        dtype,
        temperature,
        queries=((1.0, 0.25),),
        references=((0.0, 0.5), (1.0, 0.0)),
        distance=None,
    ):
        self.require_half_backend(dtype)
        q = torch.tensor(queries, dtype=dtype, device=TEST_DEVICE, requires_grad=True)
        r = torch.tensor(
            references, dtype=dtype, device=TEST_DEVICE, requires_grad=True
        )
        loss_dict = NTXentLoss(
            temperature=temperature,
            distance=(
                DotProductSimilarity(normalize_embeddings=False)
                if distance is None
                else distance
            ),
            reducer=DoNothingReducer(),
        )(
            q,
            torch.tensor([0], device=TEST_DEVICE),
            indices_tuple=repeated_positive_pair_indices(),
            ref_emb=r,
            ref_labels=torch.tensor([0, 1], device=TEST_DEVICE),
        )
        raw_losses = self.repeated_pair_raw_losses(loss_dict)
        expected = mean_pair_oracle(
            queries, references, [(0, 0), (0, 0)], [(0, 1)], temperature
        )
        # Both positive terms are equal and positive: this mean equals AvgNonZeroReducer.
        self.assert_value_and_gradients(
            raw_losses.mean().reshape(1), q, r, expected, dtype
        )

    def test_low_temperature_preserves_value_and_embedding_gradients(self):
        # gap=.875, T=.001: true loss875 exceeds even the float64 tiny-probability floor.
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                self.single_pair(dtype, 0.001)

    def test_half_precision_hard_negative_preserves_gradients(self):
        # Finite half logits at default T; the probability-domain loss previously caps near9.7.
        self.single_pair(torch.float16, 0.07, ((1.0, 0.0),), ((-1.0, 0.0), (1.0, 0.0)))

    def test_normalized_cosine_chain_rule(self):
        # The independent tangent-space derivatives include the public normalization step.
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                q = torch.tensor(
                    [[1.0, 0.0]], dtype=dtype, device=TEST_DEVICE, requires_grad=True
                )
                r = torch.tensor(
                    [[0.0, 1.0], [0.6, 0.8]],
                    dtype=dtype,
                    device=TEST_DEVICE,
                    requires_grad=True,
                )
                loss_dict = NTXentLoss(temperature=0.001, reducer=DoNothingReducer())(
                    q,
                    torch.tensor([0], device=TEST_DEVICE),
                    indices_tuple=repeated_positive_pair_indices(),
                    ref_emb=r,
                    ref_labels=torch.tensor([0, 1], device=TEST_DEVICE),
                )
                loss = self.repeated_pair_raw_losses(loss_dict).mean()
                # The two identical positive terms have the same tangent derivative; their mean retains it.
                expected = (
                    [600.0],
                    [[0.0, -200.0]],
                    [[-1000.0, 0.0], [640.0, -480.0]],
                    -600000.0,
                )
                self.assert_value_and_gradients(loss.reshape(1), q, r, expected, dtype)

    def test_actual_distance_uses_correct_low_temperature_sign(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                q = torch.tensor(
                    [[0.0, 0.0]], dtype=dtype, device=TEST_DEVICE, requires_grad=True
                )
                r = torch.tensor(
                    [[1.0, 0.0], [0.0, 0.25]],
                    dtype=dtype,
                    device=TEST_DEVICE,
                    requires_grad=True,
                )
                loss_dict = NTXentLoss(
                    temperature=0.001,
                    distance=LpDistance(normalize_embeddings=False),
                    reducer=DoNothingReducer(),
                )(
                    q,
                    torch.tensor([0], device=TEST_DEVICE),
                    indices_tuple=repeated_positive_pair_indices(),
                    ref_emb=r,
                    ref_labels=torch.tensor([0, 1], device=TEST_DEVICE),
                )
                loss = self.repeated_pair_raw_losses(loss_dict).mean()
                # Two equal positive terms are averaged, so these analytic distance derivatives are unchanged.
                expected = (
                    [750.0],
                    [[-1000.0, 1000.0]],
                    [[1000.0, 0.0], [0.0, -1000.0]],
                    -750000.0,
                )
                self.assert_value_and_gradients(loss.reshape(1), q, r, expected, dtype)

    def test_multiple_negatives_and_temperature_derivative(self):
        query_values = ((1.0, 0.25),)
        reference_values = ((0.0, 0.5), (1.0, 0.0), (0.75, 0.25))
        for dtype in (torch.float32, torch.float64):
            for value in (0.001, 0.2):
                with self.subTest(dtype=dtype, temperature=value):
                    q = torch.tensor(
                        query_values,
                        dtype=dtype,
                        device=TEST_DEVICE,
                        requires_grad=True,
                    )
                    r = torch.tensor(
                        reference_values,
                        dtype=dtype,
                        device=TEST_DEVICE,
                        requires_grad=True,
                    )
                    temperature = torch.tensor(
                        value, dtype=dtype, device=TEST_DEVICE, requires_grad=True
                    )
                    loss = NTXentLoss(
                        temperature=temperature,
                        distance=DotProductSimilarity(normalize_embeddings=False),
                    )(
                        q,
                        torch.tensor([0], device=TEST_DEVICE),
                        ref_emb=r,
                        ref_labels=torch.tensor([0, 1, 2], device=TEST_DEVICE),
                    )
                    expected = scalar_pair_oracle(
                        query_values,
                        reference_values,
                        [(0, 0)],
                        [(0, 1), (0, 2)],
                        value,
                    )
                    self.assert_value_and_gradients(
                        loss.reshape(1), q, r, expected, dtype
                    )
                    self.assertTrue(torch.isfinite(temperature.grad))
                    self.assert_close_tensor(
                        temperature.grad.double(),
                        torch.tensor(
                            expected[3], dtype=torch.float64, device=TEST_DEVICE
                        ),
                        rtol=3e-5 if dtype == torch.float32 else 2e-12,
                        atol=3e-5 if dtype == torch.float32 else 2e-12,
                    )

    def test_anchor_mask_and_empty_negative_row_have_finite_gradients(self):
        query_values = ((1.0, 0.0), (0.0, 2.0), (1.0, 1.0))
        reference_values = (
            (0.0, 0.5),
            (0.25, 0.25),
            (0.5, 0.0),
            (1.0, 0.25),
            (0.75, 0.75),
        )
        positives = [(0, 0), (0, 1), (1, 2), (2, 3)]
        negatives = [(0, 4), (1, 4)]
        for dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                self.require_half_backend(dtype)
                q = torch.tensor(
                    query_values, dtype=dtype, device=TEST_DEVICE, requires_grad=True
                )
                r = torch.tensor(
                    reference_values,
                    dtype=dtype,
                    device=TEST_DEVICE,
                    requires_grad=True,
                )
                pair_indices = tuple(
                    torch.tensor(values, device=TEST_DEVICE)
                    for values in ([0, 0, 1, 2], [0, 1, 2, 3], [0, 1], [4, 4])
                )
                loss_dict = NTXentLoss(
                    temperature=0.25,
                    distance=DotProductSimilarity(normalize_embeddings=False),
                    reducer=DoNothingReducer(),
                )(
                    q,
                    torch.tensor([0, 1, 2], device=TEST_DEVICE),
                    indices_tuple=pair_indices,
                    ref_emb=r,
                    ref_labels=torch.tensor([0, 0, 1, 2, 3], device=TEST_DEVICE),
                )
                losses = loss_dict["loss"]["losses"]
                expected = scalar_pair_oracle(
                    query_values, reference_values, positives, negatives, 0.25
                )
                self.assert_value_and_gradients(losses, q, r, expected, dtype)
                self.assertEqual(losses[-1].item(), 0.0)
                self.assertTrue(torch.equal(q.grad[2], torch.zeros_like(q.grad[2])))
                self.assertTrue(torch.equal(r.grad[3], torch.zeros_like(r.grad[3])))

    def test_moderate_temperature_preserves_existing_result(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                self.single_pair(dtype, 0.5)
                query_values = ((1.0, 0.25),)
                reference_values = ((0.0, 0.5), (1.0, 0.0))
                q = torch.tensor(
                    query_values, dtype=dtype, device=TEST_DEVICE, requires_grad=True
                )
                r = torch.tensor(
                    reference_values,
                    dtype=dtype,
                    device=TEST_DEVICE,
                    requires_grad=True,
                )
                loss = NTXentLoss(
                    temperature=0.5,
                    distance=DotProductSimilarity(normalize_embeddings=False),
                )(
                    q,
                    torch.tensor([0], device=TEST_DEVICE),
                    indices_tuple=repeated_positive_pair_indices(),
                    ref_emb=r,
                    ref_labels=torch.tensor([0, 1], device=TEST_DEVICE),
                )
                expected = mean_pair_oracle(
                    query_values, reference_values, [(0, 0), (0, 0)], [(0, 1)], 0.5
                )
                self.assert_value_and_gradients(loss.reshape(1), q, r, expected, dtype)

    def test_large_common_logit_offset_preserves_log_two_and_gradients(self):
        # Add a constant similarity shift while keeping the distance Jacobian
        # well conditioned, so this control isolates loss translation invariance.
        query_values = ((1.0, 0.0),)
        reference_values = ((0.0, 1.0), (0.0, -1.0))
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                distance = OffsetDotProductSimilarity(normalize_embeddings=False)
                actual_scores = distance(
                    torch.tensor(query_values, dtype=dtype, device=TEST_DEVICE),
                    torch.tensor(reference_values, dtype=dtype, device=TEST_DEVICE),
                )
                self.assertTrue(
                    torch.equal(
                        actual_scores,
                        torch.full(
                            (1, 2), 100000000.0, dtype=dtype, device=TEST_DEVICE
                        ),
                    )
                )
                actual_pairwise = distance.pairwise_distance(
                    torch.tensor(query_values, dtype=dtype, device=TEST_DEVICE).repeat(
                        2, 1
                    ),
                    torch.tensor(reference_values, dtype=dtype, device=TEST_DEVICE),
                )
                self.assertTrue(torch.equal(actual_scores.reshape(-1), actual_pairwise))
                self.single_pair(
                    dtype, 0.07, query_values, reference_values, distance=distance
                )

    def test_half_negative_gap_overflow_keeps_zero_gradients_finite(self):
        self.require_half_backend(torch.float16)
        # Both logits are finite, but their strongly negative difference overflows half.
        query_values = ((1.0, 0.0),)
        reference_values = ((60000.0, 0.0), (-60000.0, 0.0))
        q = torch.tensor(
            query_values, dtype=torch.float16, device=TEST_DEVICE, requires_grad=True
        )
        r = torch.tensor(
            reference_values,
            dtype=torch.float16,
            device=TEST_DEVICE,
            requires_grad=True,
        )
        loss_dict = NTXentLoss(
            temperature=1.0,
            distance=DotProductSimilarity(normalize_embeddings=False),
            reducer=DoNothingReducer(),
        )(
            q,
            torch.tensor([0], device=TEST_DEVICE),
            indices_tuple=repeated_positive_pair_indices(),
            ref_emb=r,
            ref_labels=torch.tensor([0, 1], device=TEST_DEVICE),
        )
        # Preserve both actual pair graphs even when the normal zero-loss reducer would drop them.
        losses = self.repeated_pair_raw_losses(loss_dict)
        expected = scalar_pair_oracle(
            query_values, reference_values, [(0, 0), (0, 0)], [(0, 1)], 1.0
        )
        self.assert_value_and_gradients(losses, q, r, expected, torch.float16)
        self.assertTrue(torch.equal(losses, torch.zeros_like(losses)))
        self.assertTrue(torch.equal(q.grad, torch.zeros_like(q.grad)))
        self.assertTrue(torch.equal(r.grad, torch.zeros_like(r.grad)))
