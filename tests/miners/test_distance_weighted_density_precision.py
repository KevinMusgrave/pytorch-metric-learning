import math

import pytest
import torch

from pytorch_metric_learning import miners

from .. import TEST_DEVICE


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("dimension", [128, 131072])
@pytest.mark.parametrize("precomputed", [False, True])
def test_distance_weighted_miner_retains_eligible_anchors_before_density_normalization(
    dtype, dimension, precomputed
):
    query = torch.zeros(4, dimension, device=TEST_DEVICE, dtype=dtype)
    query[:, :2] = query.new_tensor(
        [[math.cos(angle), math.sin(angle)] for angle in [0, 0.1, 0.2, 0.3]]
    )
    labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
    matrix = (
        torch.full((4, 4), 0.5, device=TEST_DEVICE, dtype=dtype)
        if precomputed
        else None
    )
    anchors, positives, negatives = miners.DistanceWeightedMiner()(
        query, labels, mat=matrix
    )
    torch.testing.assert_close(
        torch.sort(anchors).values, torch.arange(4, device=TEST_DEVICE)
    )
    assert torch.equal(labels[anchors], labels[positives])
    assert (anchors != positives).all()
    assert (labels[anchors] != labels[negatives]).all()
    assert all(index.dtype == torch.long for index in (anchors, positives, negatives))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_distance_weighted_density_keeps_rectangular_candidate_restrictions(dtype):
    query = torch.zeros(2, 131072, device=TEST_DEVICE, dtype=dtype)
    reference = torch.zeros(4, 131072, device=TEST_DEVICE, dtype=dtype)
    labels = torch.tensor([0, 1], device=TEST_DEVICE)
    ref_labels = torch.tensor([0, 0, 1, 2], device=TEST_DEVICE)
    matrix = torch.full((2, 4), 0.5, device=TEST_DEVICE, dtype=dtype)
    pairs = tuple(torch.tensor(x, device=TEST_DEVICE) for x in [[0], [1], [0], [3]])
    anchors, positives, negatives = miners.DistanceWeightedMiner()(
        query, labels, pairs, reference, ref_labels, mat=matrix
    )
    torch.testing.assert_close(anchors, torch.tensor([0], device=TEST_DEVICE))
    torch.testing.assert_close(positives, torch.tensor([1], device=TEST_DEVICE))
    torch.testing.assert_close(negatives, torch.tensor([3], device=TEST_DEVICE))


def test_distance_weighted_density_preserves_strict_cutoff_and_empty_rows():
    query = torch.zeros(2, 131072, device=TEST_DEVICE, dtype=torch.float16)
    reference = torch.zeros(3, 131072, device=TEST_DEVICE, dtype=torch.float16)
    matrix = torch.full((2, 3), 0.5, device=TEST_DEVICE, dtype=torch.float16)
    matrix[:, 2] = 1.5
    output = miners.DistanceWeightedMiner(nonzero_loss_cutoff=1.5)(
        query,
        torch.tensor([0, 0], device=TEST_DEVICE),
        ref_emb=reference,
        ref_labels=torch.tensor([0, 0, 1], device=TEST_DEVICE),
        mat=matrix,
    )
    assert all(len(indices) == 0 for indices in output)
