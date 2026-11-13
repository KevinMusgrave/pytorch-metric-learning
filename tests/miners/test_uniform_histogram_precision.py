import math
from collections import Counter

import pytest
import torch

from pytorch_metric_learning.miners import UniformHistogramMiner
from pytorch_metric_learning.types import PairIndices

from .. import TEST_DEVICE

CASES = [
    (torch.float16, [0.0, 1e-5, 2e-5], 3),
    (torch.float16, [0.0, 0.69970703125, 0.7001953125, 1.0], 100),
    (torch.bfloat16, [0.0, 0.69921875, 0.703125, 1.0], 100),
    (torch.float32, [0.0, 1e-40, 2e-40], 3),
    (torch.float64, [0.0, 1e-310, 2e-310], 3),
]


def expected_bins(distances, count):
    values = distances.tolist()
    low, high = min(values), max(values)
    if low == high:
        return [0] * len(values)
    return [
        min(math.floor((value - low) / (high - low) * count), count - 1)
        for value in values
    ]


@pytest.mark.parametrize("dtype,values,num_bins", CASES)
@pytest.mark.parametrize("positive", [True, False])
def test_occupied_bins_follow_the_observed_range(dtype, values, num_bins, positive):
    matrix = torch.tensor([values], dtype=dtype, device=TEST_DEVICE)
    embeddings = torch.ones(1, 2, dtype=dtype, device=TEST_DEVICE)
    reference = torch.ones(len(values), 2, dtype=dtype, device=TEST_DEVICE)
    labels = torch.tensor([0], device=TEST_DEVICE)
    ref_labels = torch.zeros(len(values), dtype=torch.long, device=TEST_DEVICE) + int(
        not positive
    )
    per_bin = 3
    miner = UniformHistogramMiner(
        num_bins=num_bins, pos_per_bin=per_bin, neg_per_bin=per_bin
    )
    a1, p, a2, n = miner(
        embeddings, labels, ref_emb=reference, ref_labels=ref_labels, mat=matrix
    )
    anchors, others = (a1, p) if positive else (a2, n)
    bins = expected_bins(matrix[0], num_bins)
    actual = Counter(bins[index] for index in others.tolist())
    assert actual == dict.fromkeys(set(bins), per_bin)
    assert torch.equal(anchors, torch.zeros_like(anchors))
    assert (a2 if positive else a1).numel() == 0


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_constant_candidates_keep_one_bin_and_respect_supplied_pairs(dtype):
    embeddings = torch.ones(1, 2, dtype=dtype, device=TEST_DEVICE)
    references = torch.ones(4, 2, dtype=dtype, device=TEST_DEVICE)
    labels = torch.tensor([0], device=TEST_DEVICE)
    ref_labels = torch.zeros(4, dtype=torch.long, device=TEST_DEVICE)
    candidates = PairIndices(
        torch.tensor([0, 0], device=TEST_DEVICE),
        torch.tensor([1, 3], device=TEST_DEVICE),
        torch.empty(0, dtype=torch.long, device=TEST_DEVICE),
        torch.empty(0, dtype=torch.long, device=TEST_DEVICE),
    )
    matrix = torch.ones(1, 4, dtype=dtype, device=TEST_DEVICE)
    a, p, b, n = UniformHistogramMiner(pos_per_bin=5)(
        embeddings,
        labels,
        candidates,
        ref_emb=references,
        ref_labels=ref_labels,
        mat=matrix,
    )
    assert len(a) == len(p) == 5
    assert set(p.tolist()) <= {1, 3}
    assert b.numel() == n.numel() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ordinary_full_precision_ranges_cover_each_bin(dtype):
    values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=dtype, device=TEST_DEVICE)
    anchors = torch.zeros(len(values), dtype=torch.long, device=TEST_DEVICE)
    others = torch.arange(len(values), device=TEST_DEVICE)
    _, chosen = UniformHistogramMiner(num_bins=5).uniformly_distributed_pairs(
        values, anchors, others, 2
    )
    assert Counter(chosen.tolist()) == dict.fromkeys(range(len(values)), 2)
