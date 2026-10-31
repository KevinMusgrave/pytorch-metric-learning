import pytest
import torch

from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import (
    BaseDistance,
    CosineSimilarity,
    DotProductSimilarity,
    LpDistance,
)
from pytorch_metric_learning.evaluation import CustomKNN, MatchFinder

from .. import TEST_DEVICE

LOW_PRECISION = [torch.float16, torch.bfloat16]
POINTS = [[-150.0, 0.0], [150.0, 0.0], [-150.0, 225.0], [150.0, 225.0]]
LABELS = [0, 0, 1, 1]


def squared():
    return LpDistance(normalize_embeddings=False, power=2)


def points(dtype):
    return torch.tensor(POINTS, dtype=dtype, device=TEST_DEVICE)


UNBOUNDED = {
    "squared-lp": squared,
    "lp": lambda: LpDistance(normalize_embeddings=False),
    "dot": DotProductSimilarity,
    "squared-dot": lambda: DotProductSimilarity(power=2),
}
LOSSES = {
    "ContrastiveLoss": lambda: losses.ContrastiveLoss(
        distance=squared(), neg_margin=1e5
    ),
    "GeneralizedLiftedStructureLoss": lambda: losses.GeneralizedLiftedStructureLoss(
        distance=squared()
    ),
    "MarginLoss": lambda: losses.MarginLoss(distance=squared()),
    "MultiSimilarityLoss": lambda: losses.MultiSimilarityLoss(
        distance=DotProductSimilarity()
    ),
    "NCALoss": lambda: losses.NCALoss(distance=squared()),
    "NPairsLoss": lambda: losses.NPairsLoss(distance=DotProductSimilarity()),
    "TripletMarginLoss": lambda: losses.TripletMarginLoss(distance=squared()),
}


@pytest.mark.parametrize("dtype", LOW_PRECISION)
@pytest.mark.parametrize("name", UNBOUNDED)
def test_unbounded_values_match_float64(name, dtype):
    distance = UNBOUNDED[name]()
    x = points(dtype)
    wide = x.double()
    mat = distance(x)
    assert mat.dtype == torch.float32
    assert mat.isfinite().all()
    torch.testing.assert_close(mat.double(), distance(wide), rtol=1e-5, atol=1e-3)
    reference = x.flip(0)
    pairwise = distance.pairwise(x, reference)
    assert pairwise.dtype == torch.float32
    torch.testing.assert_close(
        pairwise.double(),
        distance.pairwise(wide, wide.flip(0)),
        rtol=1e-5,
        atol=1e-3,
    )


@pytest.mark.parametrize("dtype", LOW_PRECISION)
@pytest.mark.parametrize(
    "factory",
    [LpDistance, CosineSimilarity, lambda: LpDistance(power=2)],
    ids=["lp", "cosine", "squared-lp"],
)
def test_bounded_distances_keep_their_dtype(factory, dtype):
    distance = factory()
    x = points(dtype)
    assert distance(x).dtype == dtype
    assert distance.pairwise(x, x.flip(0)).dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_higher_precision_inputs_are_unchanged(dtype):
    for factory in UNBOUNDED.values():
        distance = factory()
        x = points(dtype)
        assert distance(x).dtype == dtype


def test_custom_unnormalized_distance_is_widened():
    seen = []

    class Recording(BaseDistance):
        def compute_mat(self, query_emb, ref_emb):
            seen.append(query_emb.dtype)
            return query_emb @ ref_emb.T

    distance = Recording(normalize_embeddings=False)
    assert distance(points(torch.float16)).max() == 73125
    assert seen == [torch.float32]


@pytest.mark.parametrize("name", UNBOUNDED)
def test_autocast_does_not_narrow_the_result(name):
    distance = UNBOUNDED[name]()
    x = points(torch.float32)
    device = TEST_DEVICE.type
    with torch.autocast(device, dtype=torch.bfloat16):
        mat = distance(x)
    assert mat.dtype == torch.float32
    torch.testing.assert_close(mat.double(), distance(x.double()), rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("dtype", LOW_PRECISION)
@pytest.mark.parametrize("name", LOSSES)
def test_losses_match_float64(name, dtype):
    labels = torch.tensor(LABELS, device=TEST_DEVICE)
    embeddings = points(dtype).requires_grad_()
    wide = embeddings.detach().double().requires_grad_()
    actual = LOSSES[name]().to(TEST_DEVICE, dtype)
    expected = LOSSES[name]().to(TEST_DEVICE, torch.float64)
    loss = actual(embeddings, labels)
    reference = expected(wide, labels)
    assert loss.isfinite()
    torch.testing.assert_close(loss.double(), reference, rtol=1e-3, atol=1e-3)
    loss.backward()
    reference.backward()
    assert embeddings.grad is not None and wide.grad is not None
    assert embeddings.grad.isfinite().all()
    torch.testing.assert_close(
        embeddings.grad.double(), wide.grad, rtol=1e-2, atol=2e-2
    )


def test_multilabel_loss_keeps_the_distance_dtype():
    labels = torch.eye(2, device=TEST_DEVICE)[torch.tensor(LABELS)].bool()
    loss_func = losses.MultiSupConLoss(distance=DotProductSimilarity())
    loss = loss_func(points(torch.float16), labels)
    reference = loss_func(points(torch.float64), labels)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss.double(), reference, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [None, 1, 16])
def test_knn_keeps_finite_large_distances(batch_size):
    x = points(torch.float16)
    knn = CustomKNN(squared(), batch_size)
    values, indices = knn(x, 3, x)
    wide_values, wide_indices = CustomKNN(squared())(x.double(), 3, x.double())
    assert values.dtype == torch.float32
    assert values.isfinite().all()
    assert torch.equal(indices.sort(dim=1).values, wide_indices.sort(dim=1).values)
    torch.testing.assert_close(values.double(), wide_values, rtol=1e-5, atol=1e-3)


def test_knn_with_no_queries_keeps_shape():
    x = points(torch.float16)
    values, indices = CustomKNN(squared(), 2)(x[:0], 2, x)
    assert values.shape == (0, 2) and indices.shape == (0, 2)


def test_threshold_above_the_float16_range_matches():
    x = points(torch.float16)
    finder = MatchFinder(squared(), threshold=100000.0)
    neighbors = x[torch.tensor([1, 0, 3, 2], device=TEST_DEVICE)]
    assert finder.is_match(x, neighbors).all()
