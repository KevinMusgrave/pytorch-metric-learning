from decimal import Decimal, localcontext

import pytest
import torch

from pytorch_metric_learning import losses

from .. import TEST_DEVICE


def decimal_pnp_reference(matrix, variant, temperature=0.01, b=2, alpha=8):
    with localcontext(prec=100):
        values = [[Decimal.from_float(x) for x in row] for row in matrix]
        gradient = [[Decimal(0) for _ in matrix] for _ in matrix]
        total = Decimal(0)
        temperature = Decimal.from_float(temperature)
        b = Decimal(b)
        for anchor in range(4):
            for positive in range(4):
                if anchor // 2 != positive // 2:
                    continue
                negatives = [other for other in range(4) if anchor // 2 != other // 2]
                probabilities = [
                    1
                    / (
                        1
                        + (
                            -(values[anchor][negative] - values[anchor][positive])
                            / temperature
                        ).exp()
                    )
                    for negative in negatives
                ]
                rank = sum(probabilities, start=Decimal(0))
                logarithm = (1 + rank).ln()
                if variant == "Ds":
                    value, slope = logarithm, 1 / (1 + rank)
                elif variant == "Dq":
                    value, slope = (
                        1 - (1 + rank) ** -alpha,
                        alpha * (1 + rank) ** (-alpha - 1),
                    )
                elif variant == "Iu":
                    value, slope = (1 + rank) * logarithm, 1 + logarithm
                elif variant == "Ib":
                    value, slope = (
                        (b * rank - (1 + b * rank).ln()) / b**2,
                        rank / (1 + b * rank),
                    )
                else:
                    value, slope = rank, Decimal(1)
                total += value / 8
                for negative, probability in zip(negatives, probabilities, strict=True):
                    derivative = (
                        slope * probability * (1 - probability) / (temperature * 8)
                    )
                    gradient[anchor][negative] += derivative
                    gradient[anchor][positive] -= derivative
        return float(total), [[float(x) for x in row] for row in gradient]


def similarities(dtype, negative):
    return torch.tensor(
        [
            [1, 0.95, negative, negative],
            [0.95, 1, negative, negative],
            [negative, negative, 1, 0.95],
            [negative, negative, 0.95, 1],
        ],
        dtype=dtype,
        device=TEST_DEVICE,
        requires_grad=True,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("variant", ["Ds", "Dq", "Iu", "Ib", "O"])
@pytest.mark.parametrize("negative", [0.65, 0.8, 0.93])
def test_pnp_small_penalties_match_decimal_values_and_gradients(
    dtype, variant, negative
):
    matrix = similarities(dtype, negative)
    expected, expected_grad = decimal_pnp_reference(
        matrix.detach().double().cpu().tolist(), variant
    )
    query = torch.zeros(4, 2, dtype=dtype, device=TEST_DEVICE)
    actual = losses.PNPLoss(variant=variant, alpha=8)(
        query, torch.tensor([0, 0, 1, 1], device=TEST_DEVICE), mat=matrix
    )
    actual_grad = torch.autograd.grad(actual, matrix)[0]
    tolerance = 8e-6 if dtype == torch.float32 else 2e-10
    assert actual.dtype == dtype
    assert actual.item() > 0
    torch.testing.assert_close(
        actual, actual.new_tensor(expected), rtol=tolerance, atol=0
    )
    torch.testing.assert_close(
        actual_grad, actual_grad.new_tensor(expected_grad), rtol=tolerance, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("variant", ["Ds", "Dq", "Iu", "Ib", "O"])
def test_pnp_representable_low_precision_penalties_keep_dtype_and_gradients(
    dtype, variant
):
    matrix = similarities(dtype, 0.9)
    expected, expected_grad = decimal_pnp_reference(
        matrix.detach().double().cpu().tolist(), variant
    )
    query = torch.zeros(4, 2, dtype=dtype, device=TEST_DEVICE)
    actual = losses.PNPLoss(variant=variant, alpha=8)(
        query, torch.tensor([0, 0, 1, 1], device=TEST_DEVICE), mat=matrix
    )
    actual_grad = torch.autograd.grad(actual, matrix)[0]
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, actual.new_tensor(expected), rtol=4e-3, atol=0)
    torch.testing.assert_close(
        actual_grad, actual_grad.new_tensor(expected_grad), rtol=4e-3, atol=0
    )


@pytest.mark.parametrize("variant", ["Ds", "Dq", "Iu", "Ib", "O"])
def test_pnp_variants_keep_zero_when_no_negative_classes_exist(variant):
    query = torch.tensor(
        [[1.0, 0.1], [1.0, -0.1]],
        dtype=torch.float64,
        device=TEST_DEVICE,
        requires_grad=True,
    )
    actual = losses.PNPLoss(variant=variant, alpha=8)(
        query, torch.zeros(2, dtype=torch.long, device=TEST_DEVICE)
    )
    actual.backward()
    assert actual.item() == 0
    torch.testing.assert_close(query.grad, torch.zeros_like(query))


def test_pnp_ib_large_boundary_does_not_overflow_an_unused_series():
    matrix = similarities(torch.float32, 0.9)
    expected, expected_grad = decimal_pnp_reference(
        matrix.detach().double().cpu().tolist(), "Ib", b=1e10
    )
    actual = losses.PNPLoss(variant="Ib", b=1e10)(
        torch.zeros(4, 2, device=TEST_DEVICE),
        torch.tensor([0, 0, 1, 1], device=TEST_DEVICE),
        mat=matrix,
    )
    actual_grad = torch.autograd.grad(actual, matrix)[0]
    torch.testing.assert_close(actual, actual.new_tensor(expected), rtol=8e-6, atol=0)
    torch.testing.assert_close(
        actual_grad, actual_grad.new_tensor(expected_grad), rtol=8e-6, atol=0
    )
