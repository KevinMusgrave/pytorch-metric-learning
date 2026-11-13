import pytest
import torch

from pytorch_metric_learning.evaluation import FaissKNN

from .. import TEST_DEVICE


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("layout", ["columns", "transpose"])
@pytest.mark.parametrize("operation", ["train", "add"])
def test_persistent_indices_accept_noncontiguous_views(dtype, layout, operation):
    if layout == "columns":
        rows = torch.arange(48, dtype=dtype, device=TEST_DEVICE).reshape(8, 6)[:, ::2]
    else:
        rows = torch.arange(24, dtype=dtype, device=TEST_DEVICE).reshape(3, 8).T
    assert not rows.is_contiguous()
    original = rows.clone()
    knn = FaissKNN(reset_before=False, reset_after=False)
    if operation == "train":
        knn.train(rows)
    else:
        knn.train(rows[:4].contiguous())
        knn.add(rows[4:])
    distances, indices = knn(rows, 1)
    torch.testing.assert_close(distances, torch.zeros(8, 1, device=TEST_DEVICE))
    torch.testing.assert_close(indices[:, 0], torch.arange(8, device=TEST_DEVICE))
    assert knn.index is not None
    assert knn.index.ntotal == len(rows)
    assert torch.equal(rows, original)


def test_noncontiguous_appends_survive_save_and_load(tmp_path):
    rows = torch.arange(36.0, device=TEST_DEVICE).reshape(6, 6)[:, ::2]
    original = FaissKNN(reset_before=False, reset_after=False)
    original.train(rows[:2].contiguous())
    original.add(rows[2:])
    path = str(tmp_path / "reference.index")
    original.save(path)
    restored = FaissKNN(reset_before=False, reset_after=False)
    restored.load(path)
    distances, indices = restored(rows, 1)
    torch.testing.assert_close(distances, torch.zeros(6, 1, device=TEST_DEVICE))
    torch.testing.assert_close(indices[:, 0], torch.arange(6, device=TEST_DEVICE))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_contiguous_persistent_index_keeps_exact_neighbors(dtype):
    rows = torch.arange(24, dtype=dtype, device=TEST_DEVICE).reshape(8, 3)
    knn = FaissKNN(reset_before=False, reset_after=False)
    knn.train(rows[:4])
    knn.add(rows[4:])
    distances, indices = knn(rows, 1)
    torch.testing.assert_close(distances, torch.zeros(8, 1, device=TEST_DEVICE))
    torch.testing.assert_close(indices[:, 0], torch.arange(8, device=TEST_DEVICE))
