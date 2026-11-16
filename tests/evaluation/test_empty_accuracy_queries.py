import math

import pytest
import torch

from pytorch_metric_learning import evaluation

from .. import TEST_DEVICE


def same_columns(left, right):
    return (left == right).all(dim=-1)


@pytest.mark.parametrize("gallery", ["self", "empty", "nonempty"])
@pytest.mark.parametrize("label_mode", ["default", "custom", "columns"])
@pytest.mark.parametrize("average", [False, True])
@pytest.mark.parametrize("k", [None, 1, "max_bin_count"])
def test_empty_accuracy_queries_keep_nan_and_empty_class_reductions(
    gallery, label_mode, average, k
):
    query = torch.empty(0, 3, device=TEST_DEVICE, requires_grad=True)
    labels = torch.empty(
        (0, 2) if label_mode == "columns" else (0,),
        dtype=torch.long,
        device=TEST_DEVICE,
    )
    comparison = same_columns if label_mode == "columns" else torch.eq
    calculator = evaluation.AccuracyCalculator(
        k=k,
        avg_of_avgs=average,
        label_comparison_fn=None if label_mode == "default" else comparison,
        custom_metrics={
            "ndcg": evaluation.ndcg_metric(
                lambda left, right: comparison(left, right).double()
            )
        },
    )
    reference = None
    ref_labels = None
    if gallery != "self":
        count = 0 if gallery == "empty" else 3
        reference = torch.eye(3, device=TEST_DEVICE)[:count]
        ref_labels = torch.arange(count, device=TEST_DEVICE)
        if label_mode == "columns":
            ref_labels = torch.stack((ref_labels, ref_labels), dim=1)
    pooled = calculator.get_accuracy(
        query, labels, reference=reference, reference_labels=ref_labels
    )
    per_class = calculator.get_accuracy_per_class(
        query, labels, reference=reference, reference_labels=ref_labels
    )
    assert set(pooled) == set(calculator.metrics)
    assert all(math.isnan(value) for value in pooled.values())
    assert per_class == {metric: {} for metric in calculator.metrics}
    assert calculator.k == k
    assert query.grad is None


def test_empty_accuracy_does_not_invoke_nearest_neighbor_backend():
    calls = []

    def backend(query, k, reference, ref_includes_query):
        calls.append((len(query), k, len(reference), ref_includes_query))
        raise AssertionError("empty queries need no search")

    calculator = evaluation.AccuracyCalculator(
        include=("precision_at_1",), knn_func=backend
    )
    result = calculator.get_accuracy(
        torch.empty(0, 2), torch.empty(0, dtype=torch.long)
    )
    assert math.isnan(result["precision_at_1"])
    assert calls == []


@pytest.mark.parametrize("self_search", [False, True])
def test_standalone_ndcg_returns_one_value_per_empty_query(self_search):
    labels = torch.empty(0, dtype=torch.long, device=TEST_DEVICE)
    inputs = evaluation.RetrievalInputs(
        labels,
        labels.reshape(0, 0),
        torch.empty(0, 0, dtype=torch.bool, device=TEST_DEVICE),
        labels,
        labels,
        self_search,
    )
    result = evaluation.ndcg(inputs, lambda left, right: (left == right).double())
    assert result.shape == (0,)
    assert result.dtype == torch.float64
    assert result.device == labels.device


def test_empty_queries_still_validate_label_lengths_and_custom_metric_shape():
    query = torch.empty(0, 2)
    calculator = evaluation.AccuracyCalculator()
    with pytest.raises(ValueError, match="same length"):
        calculator.get_accuracy(query, torch.tensor([0]))
    calculator = evaluation.AccuracyCalculator(
        include=("bad",),
        custom_metrics={
            "bad": evaluation.RetrievalMetric(lambda inputs: torch.ones(1))
        },
    )
    with pytest.raises(ValueError, match="one value per query"):
        calculator.get_accuracy(query, torch.empty(0, dtype=torch.long))


def test_nonempty_queries_against_empty_gallery_keep_existing_nan_contract():
    calculator = evaluation.AccuracyCalculator()
    query = torch.eye(2)
    result = calculator.get_accuracy(
        query,
        torch.tensor([0, 1]),
        reference=torch.empty(0, 2),
        reference_labels=torch.empty(0, dtype=torch.long),
    )
    assert all(math.isnan(value) for value in result.values())
