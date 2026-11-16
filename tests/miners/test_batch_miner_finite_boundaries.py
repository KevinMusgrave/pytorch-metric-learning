import pytest
import torch

from pytorch_metric_learning import distances, miners
from pytorch_metric_learning.types import PairIndices

from .. import TEST_DEVICE, TEST_DTYPES


def boundary_case(dtype, case):
    distance = (
        distances.LpDistance(normalize_embeddings=False)
        if case == "distance-max"
        else distances.DotProductSimilarity(normalize_embeddings=False)
    )
    value = (
        torch.finfo(dtype).min if case == "similarity-min" else torch.finfo(dtype).max
    )
    return distance, value


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("case", ["distance-max", "similarity-max", "similarity-min"])
@pytest.mark.parametrize(
    "positive,negative",
    [
        ("easy", "hard"),
        ("hard", "easy"),
        ("hard", "hard"),
        ("all", "hard"),
        ("easy", "all"),
    ],
)
def test_batch_pair_miner_keeps_finite_boundary_candidates(
    dtype, case, positive, negative
):
    distance, value = boundary_case(dtype, case)
    x = torch.zeros(4, 2, dtype=dtype, device=TEST_DEVICE)
    labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
    matrix = torch.full((4, 4), value, dtype=dtype, device=TEST_DEVICE)
    miner = miners.BatchEasyHardMiner(
        pos_strategy=positive, neg_strategy=negative, distance=distance
    )
    anchors, positives, neg_anchors, negatives = miner(x, labels, mat=matrix)
    assert anchors.tolist() == [0, 1, 2, 3]
    assert positives.tolist() == [1, 0, 3, 2]
    assert torch.equal(labels[anchors], labels[positives])
    assert torch.all(labels[neg_anchors] != labels[negatives])
    if negative != "all":
        assert neg_anchors.tolist() == [0, 1, 2, 3]
        assert negatives.tolist() == [2, 2, 0, 0]
    torch.testing.assert_close(matrix, torch.full_like(matrix, value))


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("case", ["distance-max", "similarity-max", "similarity-min"])
def test_batch_hard_miner_keeps_finite_boundary_triplets(dtype, case):
    distance, value = boundary_case(dtype, case)
    x = torch.zeros(4, 2, dtype=dtype, device=TEST_DEVICE)
    labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
    matrix = torch.full((4, 4), value, dtype=dtype, device=TEST_DEVICE)
    result = miners.BatchHardMiner(distance=distance)(x, labels, mat=matrix)
    assert tuple(indices.tolist() for indices in result) == (
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 2, 0, 0],
    )


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("case", ["distance-max", "similarity-max", "similarity-min"])
def test_batch_miner_preserves_restricted_rectangular_boundary_candidates(dtype, case):
    distance, value = boundary_case(dtype, case)
    x = torch.zeros(3, 2, dtype=dtype, device=TEST_DEVICE)
    reference = torch.zeros(5, 2, dtype=dtype, device=TEST_DEVICE)
    labels = torch.tensor([0, 1, 1], device=TEST_DEVICE)
    ref_labels = torch.tensor([2, 0, 2, 2, 1], device=TEST_DEVICE)
    candidates = PairIndices(
        *(
            torch.tensor(indices, device=TEST_DEVICE)
            for indices in ([0, 2], [1, 4], [0, 2], [3, 0])
        )
    )
    matrix = torch.full((3, 5), value, dtype=dtype, device=TEST_DEVICE)
    miner = miners.BatchEasyHardMiner(
        pos_strategy="easy",
        neg_strategy="hard",
        allowed_pos_range=(value, value),
        allowed_neg_range=(value, value),
        distance=distance,
    )
    result = miner(x, labels, candidates, reference, ref_labels, mat=matrix)
    for actual, expected in zip(result, candidates, strict=True):
        torch.testing.assert_close(actual, expected)
