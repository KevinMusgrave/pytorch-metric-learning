import math

import pytest
import torch

from pytorch_metric_learning import distances, losses, reducers

from .. import TEST_DEVICE


def normalized_log_probability(
    matrix, labels, references, scale, inverted, self_search
):
    scores = matrix if inverted else -matrix
    terms = []
    for row, label in enumerate(labels):
        eligible = torch.arange(len(references), device=matrix.device)
        if self_search:
            eligible = eligible[eligible != row]
        positive = references[eligible] == label
        if positive.any():
            log_probabilities = torch.log_softmax(scores[row, eligible] * scale, dim=0)
            terms.append(-torch.logsumexp(log_probabilities[positive], dim=0))
    return torch.stack(terms)


@pytest.mark.parametrize("dtype,offset", [(torch.float32, 1e8), (torch.float64, 1e16)])
@pytest.mark.parametrize("inverted", [False, True])
@pytest.mark.parametrize("self_search", [False, True])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_nca_large_row_offsets_preserve_losses_and_matrix_gradients(
    dtype, offset, inverted, self_search, reduction
):
    labels = torch.tensor([0, 0, 1, 1, 2], device=TEST_DEVICE)
    ref_labels = (
        labels if self_search else torch.tensor([0, 1, 1, 3], device=TEST_DEVICE)
    )
    values = (
        torch.arange(len(labels) * len(ref_labels), device=TEST_DEVICE).reshape(
            len(labels), len(ref_labels)
        )
        % 4
        * 8
    )
    matrix = (values.to(dtype) + offset).requires_grad_()
    reference = matrix.detach().clone().requires_grad_()
    query = torch.zeros(len(labels), 2, dtype=dtype, device=TEST_DEVICE)
    distance = (
        distances.DotProductSimilarity(normalize_embeddings=False)
        if inverted
        else distances.LpDistance(normalize_embeddings=False)
    )
    reducer = reducers.SumReducer() if reduction == "sum" else reducers.MeanReducer()
    function = losses.NCALoss(distance=distance, reducer=reducer)
    args = (
        {}
        if self_search
        else {
            "ref_emb": torch.zeros(len(ref_labels), 2, dtype=dtype, device=TEST_DEVICE),
            "ref_labels": ref_labels,
        }
    )
    actual = function(query, labels, mat=matrix, **args)
    terms = normalized_log_probability(
        reference, labels, ref_labels, 1, inverted, self_search
    )
    expected = terms.sum() if reduction == "sum" else terms.mean()
    actual_grad = torch.autograd.grad(actual, matrix)[0]
    expected_grad = torch.autograd.grad(expected, reference)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(matrix.detach(), reference.detach())


@pytest.mark.parametrize(
    "dtype,magnitude", [(torch.float32, 10000.0), (torch.float64, 1e8)]
)
def test_nca_simplex_embeddings_preserve_nonzero_training_objective(dtype, magnitude):
    query = (torch.eye(4, dtype=dtype, device=TEST_DEVICE) * magnitude).requires_grad_()
    labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
    function = losses.NCALoss(
        distance=distances.LpDistance(normalize_embeddings=False, power=2)
    )
    actual = function(query, labels)
    actual_grad = torch.autograd.grad(actual, query)[0]
    expected_grad = torch.full_like(query, magnitude / 3)
    expected_grad.fill_diagonal_(0)
    expected_grad[
        torch.arange(4, device=TEST_DEVICE),
        torch.tensor([1, 0, 3, 2], device=TEST_DEVICE),
    ] = -2 * magnitude / 3
    torch.testing.assert_close(
        actual, actual.new_tensor(math.log(3)), rtol=2e-6, atol=1e-7
    )
    torch.testing.assert_close(
        actual_grad / magnitude, expected_grad / magnitude, rtol=2e-6, atol=2e-7
    )


@pytest.mark.parametrize(
    "dtype,magnitude", [(torch.float32, 10000.0), (torch.float64, 1e8)]
)
def test_proxy_nca_equal_distant_centers_preserve_training_gradients(dtype, magnitude):
    query = torch.zeros(1, 3, dtype=dtype, device=TEST_DEVICE, requires_grad=True)
    function = losses.ProxyNCALoss(
        3, 3, distance=distances.LpDistance(normalize_embeddings=False, power=2)
    ).to(device=TEST_DEVICE, dtype=dtype)
    with torch.no_grad():
        function.weight.copy_(torch.eye(3, dtype=dtype, device=TEST_DEVICE) * magnitude)
    actual = function(query, torch.tensor([0], device=TEST_DEVICE))
    actual_grad, weight_grad = torch.autograd.grad(actual, (query, function.weight))
    expected_grad = query.new_tensor([[-4 / 3, 2 / 3, 2 / 3]])
    expected_weight = torch.diag(query.new_tensor([4 / 3, -2 / 3, -2 / 3]))
    torch.testing.assert_close(
        actual, actual.new_tensor(math.log(3)), rtol=2e-6, atol=1e-7
    )
    torch.testing.assert_close(
        actual_grad / magnitude, expected_grad, rtol=2e-6, atol=2e-7
    )
    torch.testing.assert_close(
        weight_grad / magnitude, expected_weight, rtol=2e-6, atol=2e-7
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(0, 0), (1, 1), (3, 0), (3, 2)])
def test_nca_without_eligible_positive_rows_remains_differentiable_zero(dtype, shape):
    rows, columns = shape
    query = torch.zeros(rows, 2, dtype=dtype, device=TEST_DEVICE, requires_grad=True)
    labels = torch.arange(rows, device=TEST_DEVICE)
    reference = torch.zeros(columns, 2, dtype=dtype, device=TEST_DEVICE)
    ref_labels = torch.arange(columns, device=TEST_DEVICE) + rows
    actual = losses.NCALoss()(query, labels, ref_emb=reference, ref_labels=ref_labels)
    actual.backward()
    assert actual.item() == 0
    torch.testing.assert_close(query.grad, torch.zeros_like(query))
