from decimal import Decimal, localcontext

import pytest
import torch

from pytorch_metric_learning import losses

from .. import TEST_DEVICE


def decimal_margins(counts, power, bounds):
    with localcontext(prec=400):
        exponent = Decimal.from_float(power)
        ratios = [
            (-exponent * (Decimal(value) / min(counts)).ln()).exp() for value in counts
        ]
        minimum = min(ratios)
        lower, upper = map(Decimal.from_float, bounds)
        return [
            float(lower + (upper - lower) * (value - minimum) / (1 - minimum))
            for value in ratios
        ]


@pytest.mark.parametrize(
    "counts",
    [
        [10, 100, 1000],
        [1, 2, 4],
        [2**60, 2**60 + 1, 2**60 + 2],
        [2**63 - 3, 2**63 - 2, 2**63 - 1],
    ],
)
@pytest.mark.parametrize("power", [1e-20, 0.25, 1000.0])
@pytest.mark.parametrize("bounds", [(0.0, 0.5), (0.05, 0.5)])
def test_dynamic_arcface_margins_match_decimal_across_frequency_and_power_ranges(
    counts, power, bounds
):
    model = losses.DynamicArcFaceLoss(
        counts, 2, power=power, margin_bounds=bounds, sub_centers=1
    )
    expected = model.class_margins.new_tensor(decimal_margins(counts, power, bounds))
    torch.testing.assert_close(model.class_margins, expected, rtol=2e-7, atol=1e-8)
    assert torch.isfinite(model.class_margins).all()
    assert model.class_margins[0] == bounds[1]
    assert model.class_margins[-1] == bounds[0]
    assert (model.class_margins[:-1] >= model.class_margins[1:]).all()


@pytest.mark.parametrize("power", [1e-20, 0.25, 1000.0])
def test_dynamic_arcface_frequency_scaling_and_class_permutation_keep_margins(power):
    original = losses.DynamicArcFaceLoss([10, 100, 1000], 2, power=power)
    scaled = losses.DynamicArcFaceLoss([10000, 1000, 100000], 2, power=power)
    torch.testing.assert_close(
        scaled.class_margins, original.class_margins[torch.tensor([1, 0, 2])]
    )


@pytest.mark.parametrize("power", [1e-20, 1000.0, 1e308])
def test_dynamic_arcface_extreme_finite_powers_produce_finite_training_gradients(power):
    model = losses.DynamicArcFaceLoss(
        [10, 100, 1000], 2, power=power, sub_centers=1
    ).to(TEST_DEVICE)
    with torch.no_grad():
        model.weight.copy_(model.weight.new_tensor([[1, 0], [0, 1], [-1, 0]]))
    query = torch.tensor(
        [[0.8, 0.2], [0.2, 0.8], [-0.8, 0.2]], device=TEST_DEVICE, requires_grad=True
    )
    value = model(query, torch.arange(3, device=TEST_DEVICE))
    value.backward()
    assert torch.isfinite(value)
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()


def test_dynamic_arcface_subnormal_power_keeps_logarithmic_limit():
    model = losses.DynamicArcFaceLoss([10, 100, 1000], 2, power=5e-324)
    torch.testing.assert_close(
        model.class_margins, model.class_margins.new_tensor([0.5, 0.275, 0.05])
    )


@pytest.mark.parametrize(
    "power", [float("inf"), float("-inf"), float("nan"), 0.0, -1.0]
)
def test_dynamic_arcface_rejects_nonfinite_or_nonpositive_powers(power):
    with pytest.raises(ValueError, match="power"):
        losses.DynamicArcFaceLoss([10, 100], 2, power=power)
