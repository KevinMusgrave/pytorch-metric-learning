import pytest
import torch

from pytorch_metric_learning.utils import indexing

from .. import TEST_DEVICE, TEST_DTYPES


@pytest.mark.parametrize("scale", [1e-300, 1e-50, 1.0, 1e50, 1e300])
@pytest.mark.parametrize("t_per_anchor", [None, 3])
@pytest.mark.parametrize("separate_reference", [False, True])
def test_weighted_triplet_draws_preserve_row_scale(
    scale, t_per_anchor, separate_reference
):
    labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
    reference = (
        torch.tensor([0, 1, 0, 2, 1], device=TEST_DEVICE)
        if separate_reference
        else None
    )
    reference_labels = labels if reference is None else reference
    weights = 2.0 ** (torch.arange(len(reference_labels), device=TEST_DEVICE) % 3)
    weights = weights.double().expand(len(labels), -1).clone()
    torch.manual_seed(12)
    expected = indexing.get_random_triplet_indices(
        labels, reference, t_per_anchor, weights
    )
    scaled = weights * scale
    scaled[labels[:, None] == reference_labels] = 1e308
    original = scaled.clone()
    torch.manual_seed(12)
    actual = indexing.get_random_triplet_indices(
        labels, reference, t_per_anchor, scaled
    )
    assert len(actual.anchors) > 0
    for result, correct in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, correct)
    torch.testing.assert_close(scaled, original)


@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e300])
def test_weighted_triplet_single_eligible_negative_stays_eligible(scale):
    labels = torch.tensor([0, 1], device=TEST_DEVICE)
    reference = torch.tensor([0, 1, 2], device=TEST_DEVICE)
    weights = torch.tensor(
        [[1e308, 0, scale], [scale, 1e308, 0]], dtype=torch.float64, device=TEST_DEVICE
    )
    actual = indexing.get_random_triplet_indices(labels, reference, 7, weights)
    expected = ([0] * 7 + [1] * 7, [0] * 7 + [1] * 7, [2] * 7 + [0] * 7)
    assert tuple(x.tolist() for x in actual) == expected


@pytest.mark.parametrize("shape", [(0, 4), (4, 0), (0, 0)])
def test_weighted_triplet_empty_comparison_axes(shape):
    labels = torch.zeros(shape[0], dtype=torch.long, device=TEST_DEVICE)
    reference = torch.ones(shape[1], dtype=torch.long, device=TEST_DEVICE)
    weights = torch.empty(shape, dtype=torch.float64, device=TEST_DEVICE)
    actual = indexing.get_random_triplet_indices(labels, reference, 2, weights)
    assert all(len(x) == 0 for x in actual)


@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_weighted_triplet_zero_eligible_weights_stay_empty(dtype):
    labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
    weights = (labels[:, None] == labels).to(dtype)
    actual = indexing.get_random_triplet_indices(
        labels, t_per_anchor=2, weights=weights
    )
    assert all(len(x) == 0 for x in actual)
