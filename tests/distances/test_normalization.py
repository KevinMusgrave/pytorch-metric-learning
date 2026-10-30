import pytest
import torch
from torch.nn import functional as F

from pytorch_metric_learning.distances import CosineSimilarity, LpDistance
from pytorch_metric_learning.losses import ContrastiveLoss

from .. import TEST_DEVICE
from ..helpers import assert_matches_float64

EPS = {
    torch.float16: 2.0**-14,
    torch.bfloat16: 1e-12,
    torch.float32: 1e-12,
    torch.float64: 1e-12,
}
SCALES = {torch.float16: 1e-6, torch.bfloat16: 1e-20, torch.float32: 1e-20}


def reference_normalize(x: torch.Tensor, p: float, eps: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(x, ord=p, dim=1, keepdim=True)
    return x / norms.clamp_min(eps)


def rows(dtype: torch.dtype) -> torch.Tensor:
    tiny = SCALES[dtype]
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [tiny, -2 * tiny, tiny],
            [1e-3, 2e-3, -1e-3],
            [0.5, -1.0, 2.0],
        ],
        dtype=torch.float64,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("p", [1, 2])
def test_zero_and_tiny_rows_match_float64(dtype, p):
    distance = LpDistance(p=p)
    weights = torch.tensor(
        [[0.25, -0.5, 0.75], [0.5, 0.25, -0.25], [-0.75, 0.5, 0.25], [1.0, 0.5, -0.5]],
        device=TEST_DEVICE,
    )

    def compute(x):
        normalized = distance.normalize(x)
        assert normalized.dtype == dtype
        return (normalized * weights.to(dtype)).sum()

    def reference(x):
        return (reference_normalize(x, p, EPS[dtype]) * weights.double()).sum()

    assert_matches_float64(compute, reference, [rows(dtype)], dtype)
    normalized = distance.normalize(rows(dtype).to(TEST_DEVICE, dtype))
    assert torch.equal(normalized[0], torch.zeros_like(normalized[0]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cosine_similarity_of_zero_row_is_zero_with_finite_gradients(dtype):
    def compute(x):
        return CosineSimilarity()(x).sum()

    def reference(x):
        normalized = reference_normalize(x, 2, EPS[dtype])
        return (normalized @ normalized.t()).sum()

    assert_matches_float64(compute, reference, [rows(dtype)], dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_contrastive_loss_with_zero_row_matches_float64(dtype):
    embeddings = torch.tensor(
        [[1.0, 2.0], [0.0, 0.0], [3.0, -1.0], [-1.0, 0.5]], dtype=torch.float64
    )
    labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)

    def reference(x):
        normalized = reference_normalize(x, 2, EPS[dtype])
        distances = torch.linalg.vector_norm(
            normalized[:, None] - normalized[None, :], dim=-1
        )
        same = labels[:, None] == labels[None, :]
        eye = torch.eye(len(labels), dtype=torch.bool, device=TEST_DEVICE)
        positives = distances[same & ~eye]
        negatives = F.relu(1.5 - distances[~same])
        return positives[positives > 0].mean() + negatives[negatives > 0].mean()

    def compute(x):
        return ContrastiveLoss(neg_margin=1.5)(x, labels)

    loss = assert_matches_float64(compute, reference, [embeddings], dtype)
    assert loss > 0


def test_cosine_similarity_has_no_norm_or_power_options():
    for option in ({"p": 1}, {"power": 2}):
        with pytest.raises(TypeError):
            CosineSimilarity(**option)
    distance = CosineSimilarity()
    assert (distance.p, distance.power, distance.normalize_embeddings) == (2, 1, True)
