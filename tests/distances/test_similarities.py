from typing import Any

import pytest
import torch

from pytorch_metric_learning.distances import (
    CosineSimilarity,
    DotProductSimilarity,
    SNRDistance,
)
from pytorch_metric_learning.utils import _common

from .. import TEST_DEVICE, TEST_DTYPES
from ..helpers import assert_matches_float64

QUERY = torch.tensor(
    [[0.5, -1.0, 2.0, 0.25], [1.5, 0.75, -0.5, 1.0], [-0.25, 0.5, 0.75, -2.0]],
    dtype=torch.float64,
)
REFERENCE = torch.tensor(
    [
        [1.0, 0.5, -0.75, 0.25],
        [-1.5, 2.0, 0.5, -0.5],
        [0.25, -0.25, 1.25, 1.0],
        [2.0, 1.0, 0.5, -1.5],
    ],
    dtype=torch.float64,
)
WEIGHTS = torch.tensor(
    [[0.5, -1.0, 1.5, 0.25], [2.0, 0.75, -0.5, 1.0], [-1.25, 0.5, 1.0, -0.75]],
    dtype=torch.float64,
)


def lp_norm(x, p):
    return x.abs().pow(p).sum().pow(1 / p)


def reference_matrix(entry, query, reference):
    return torch.stack([torch.stack([entry(q, r) for r in reference]) for q in query])


def cosine(q, r):
    return (q * r).sum() / (lp_norm(q, 2) * lp_norm(r, 2))


def dot_product(p, power, normalize):
    def entry(q, r):
        if normalize:
            q, r = q / lp_norm(q, p), r / lp_norm(r, p)
        return (q * r).sum() ** power

    return entry


def signal_to_noise(q, r):
    def variance(x):
        return ((x - x.mean()) ** 2).sum() / (len(x) - 1)

    q, r = q / lp_norm(q, 2), r / lp_norm(r, 2)
    return variance(q - r) / variance(q)


CASES = {
    "cosine": (CosineSimilarity(), cosine, True),
    "dot": (
        DotProductSimilarity(normalize_embeddings=False),
        dot_product(2, 1, False),
        True,
    ),
    "dot_l1_squared": (
        DotProductSimilarity(normalize_embeddings=True, p=1, power=2),
        dot_product(1, 2, True),
        True,
    ),
    "snr": (SNRDistance(), signal_to_noise, False),
}


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("case", CASES)
def test_matrix_matches_float64_reference(case, dtype):
    distance, entry, inverted = CASES[case]
    assert distance.is_inverted is inverted
    weights = WEIGHTS.to(TEST_DEVICE)

    def compute(query, reference):
        mat = distance(query, reference)
        widened = not distance.normalize_embeddings
        expected_dtype = _common.accumulation_dtype(dtype) if widened else dtype
        assert mat.shape == (3, 4) and mat.dtype == expected_dtype
        return (mat * weights.to(dtype)).sum()

    def expected(query, reference):
        return (reference_matrix(entry, query, reference) * weights).sum()

    assert_matches_float64(compute, expected, [QUERY, REFERENCE], dtype)


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("case", CASES)
def test_pairwise_matches_float64_reference(case, dtype):
    distance, entry, _ = CASES[case]
    weights = WEIGHTS[0].to(TEST_DEVICE)
    reference_rows = REFERENCE[:3]

    def compute(query, reference):
        return (distance.pairwise(query, reference) * weights[:3].to(dtype)).sum()

    def expected(query, reference):
        values = torch.stack(
            [entry(q, r) for q, r in zip(query, reference, strict=True)]
        )
        return (values * weights[:3]).sum()

    assert_matches_float64(compute, expected, [QUERY, reference_rows], dtype)


@pytest.mark.parametrize("case", CASES)
def test_gradients_match_finite_differences(case):
    distance, _, _ = CASES[case]
    query = QUERY.to(TEST_DEVICE).requires_grad_()
    reference = REFERENCE.to(TEST_DEVICE).requires_grad_()
    assert torch.autograd.gradcheck(distance, (query, reference))
    assert torch.autograd.gradcheck(distance.pairwise, (query, reference[:3]))


@pytest.mark.parametrize("case", CASES)
def test_self_comparison_uses_query_as_reference(case):
    distance, _, _ = CASES[case]
    query = QUERY.to(TEST_DEVICE)
    torch.testing.assert_close(distance(query), distance(query, query))


def test_cosine_similarity_equals_normalized_dot_product_and_has_no_options():
    query, reference = QUERY.to(TEST_DEVICE), REFERENCE.to(TEST_DEVICE)
    torch.testing.assert_close(
        CosineSimilarity()(query, reference),
        DotProductSimilarity(normalize_embeddings=True)(query, reference),
    )
    options: dict[str, Any] = {"power": 2}
    with pytest.raises(TypeError):
        CosineSimilarity(**options)


def test_dot_product_default_uses_embeddings_as_given():
    query = torch.tensor([[3.0, 4.0]], device=TEST_DEVICE)
    reference = torch.tensor([[2.0, 0.0], [0.0, -1.0]], device=TEST_DEVICE)
    distance = DotProductSimilarity()
    expected = torch.tensor([[6.0, -4.0]], device=TEST_DEVICE)
    torch.testing.assert_close(distance(query, reference), expected)
    torch.testing.assert_close(
        distance.pairwise(query.expand(2, -1), reference), expected[0]
    )


def test_similarities_reject_non_matrix_inputs():
    for distance in (CosineSimilarity(), DotProductSimilarity(), SNRDistance()):
        with pytest.raises(ValueError):
            distance(QUERY[0])
        with pytest.raises(ValueError):
            distance.pairwise(QUERY, REFERENCE)
