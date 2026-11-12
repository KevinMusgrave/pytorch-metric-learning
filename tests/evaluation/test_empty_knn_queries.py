import pytest
import torch

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.evaluation import CustomKNN, FaissKNN


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("batch_size", [None, 2])
@pytest.mark.parametrize("exclude_self", [False, True])
@pytest.mark.parametrize("k", [0, 2])
def test_empty_custom_queries_keep_shape_dtype_and_no_grad(
    dtype, normalize, batch_size, exclude_self, k
):
    query = torch.empty(0, 3, dtype=dtype, requires_grad=True)
    reference = torch.arange(12, dtype=dtype).reshape(4, 3)
    knn = CustomKNN(LpDistance(normalize_embeddings=normalize), batch_size)
    distances, indices = knn(query, k, reference, exclude_self)
    expected_dtype = (
        torch.float32
        if dtype in (torch.float16, torch.bfloat16) and not normalize
        else dtype
    )
    assert distances.shape == indices.shape == (0, k)
    assert distances.dtype == expected_dtype
    assert indices.dtype == torch.long
    assert distances.device == indices.device == query.device
    assert not distances.requires_grad


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("exclude_self", [False, True])
@pytest.mark.parametrize("k", [1, 3])
def test_empty_faiss_queries_keep_shape_and_index_state(dtype, exclude_self, k):
    query = torch.empty(0, 3, dtype=dtype)
    reference = torch.arange(12, dtype=dtype).reshape(4, 3)
    knn = FaissKNN()
    distances, indices = knn(query, k, reference, exclude_self)
    assert distances.shape == indices.shape == (0, k)
    assert distances.dtype == torch.float32
    assert indices.dtype == torch.long
    assert distances.device == indices.device == query.device
    assert knn.index is None


@pytest.mark.parametrize("batch_size", [None, 2])
@pytest.mark.parametrize("exclude_self", [False, True])
def test_empty_custom_queries_still_validate_neighbor_count(batch_size, exclude_self):
    query = torch.empty(0, 3)
    reference = torch.ones(4, 3)
    knn = CustomKNN(LpDistance(), batch_size)
    with pytest.raises(RuntimeError):
        knn(query, 5, reference, exclude_self)
