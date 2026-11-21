import pytest
import torch

from pytorch_metric_learning import distances, losses

from .. import TEST_DEVICE


def independent_ranked_mean(matrix, labels, temperature, margin, alpha, imbalance):
    rows = []
    for anchor in range(len(labels)):
        positive = []
        negative = []
        for other in range(len(labels)):
            if anchor == other:
                continue
            value = matrix[anchor, other]
            if labels[anchor] == labels[other] and value > alpha - margin:
                positive.append(value - (alpha - margin))
            elif labels[anchor] != labels[other] and value < alpha:
                negative.append(alpha - value)
        parts = []
        for selected in (positive, negative):
            if selected:
                values = torch.stack(selected)
                parts.append(
                    (torch.softmax(values * temperature, dim=0) * values).sum()
                )
            else:
                parts.append(matrix[anchor].sum() * 0)
        rows.append((1 - imbalance) * parts[0] + imbalance * parts[1])
    return torch.stack(rows).mean()


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("temperature", [100000.0, -100000.0, 0.0, 2.0])
@pytest.mark.parametrize("imbalance", [0.0, 0.5, 1.0])
def test_ranked_list_temperature_preserves_finite_values_and_matrix_gradients(
    dtype, temperature, imbalance
):
    labels = torch.tensor([0, 0, 1, 1, 2], device=TEST_DEVICE)
    values = [
        [0, 1.5, 0.5, 0.5, 1.75],
        [1.5, 0, 0.5, 0.75, 1.75],
        [0.5, 0.5, 0, 1.5, 0.75],
        [0.5, 0.75, 1.5, 0, 0.5],
        [1.75, 1.75, 0.75, 0.5, 0],
    ]
    matrix = torch.tensor(values, dtype=dtype, device=TEST_DEVICE, requires_grad=True)
    reference = matrix.detach().double().requires_grad_()
    query = torch.zeros(5, 2, dtype=dtype, device=TEST_DEVICE)
    function = losses.RankedListLoss(
        margin=0.4,
        neg_temperature=temperature,
        pos_temperature=temperature,
        imbalance=imbalance,
    )
    actual = function(query, labels, mat=matrix)
    expected = independent_ranked_mean(
        reference, labels, temperature, 0.4, 1.2, imbalance
    )
    actual_grad = torch.autograd.grad(actual, matrix)[0]
    expected_grad = torch.autograd.grad(expected, reference)[0]
    assert torch.isfinite(actual)
    assert torch.isfinite(actual_grad).all()
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=1e-7)
    tolerance = (
        8e-3 if dtype == torch.bfloat16 else 2e-3 if dtype == torch.float16 else 2e-5
    )
    torch.testing.assert_close(
        actual_grad.double(), expected_grad, rtol=tolerance, atol=1e-7
    )


@pytest.mark.parametrize("temperature", [1e32, -1e32])
def test_ranked_list_centers_finite_scores_before_large_temperature(temperature):
    labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
    matrix = torch.full(
        (4, 4), 1e7, dtype=torch.float32, device=TEST_DEVICE, requires_grad=True
    )
    reference = matrix.detach().double().requires_grad_()
    query = torch.zeros(4, 2, device=TEST_DEVICE)
    function = losses.RankedListLoss(
        margin=1,
        alpha=1,
        neg_temperature=temperature,
        pos_temperature=temperature,
        distance=distances.LpDistance(normalize_embeddings=False),
    )
    actual = function(query, labels, mat=matrix)
    expected = independent_ranked_mean(reference, labels, temperature, 1, 1, 0.5)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-7, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, matrix)[0].double(),
        torch.autograd.grad(expected, reference)[0],
        rtol=1e-7,
        atol=0,
    )


@pytest.mark.parametrize("labels", [[], [0], [0, 1, 2]])
def test_ranked_list_empty_violation_sets_keep_zero_and_finite_gradients(labels):
    query = torch.zeros(
        len(labels), 2, dtype=torch.float16, device=TEST_DEVICE, requires_grad=True
    )
    label_tensor = torch.tensor(labels, dtype=torch.long, device=TEST_DEVICE)
    function = losses.RankedListLoss(
        margin=0.4, neg_temperature=1e5, pos_temperature=1e5, alpha=-1
    )
    value = function(query, label_tensor)
    value.backward()
    assert value.item() == 0
    torch.testing.assert_close(query.grad, torch.zeros_like(query))
