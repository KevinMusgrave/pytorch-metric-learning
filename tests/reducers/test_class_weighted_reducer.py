import unittest

import pytest
import torch

from pytorch_metric_learning.reducers import ClassWeightedReducer

from .. import TEST_DEVICE, TEST_DTYPES, helpers


class TestClassWeightedReducer(unittest.TestCase):
    def test_class_weighted_reducer(self):
        torch.manual_seed(99114)
        class_weights = torch.tensor([1, 0.9, 1, 0.1, 0, 0, 0, 0, 0, 0])
        for dtype in TEST_DTYPES:
            reducer = ClassWeightedReducer(class_weights).to(TEST_DEVICE)
            batch_size = 100
            num_classes = 10
            embedding_size = 64
            embeddings = (
                torch.randn(batch_size, embedding_size).type(dtype).to(TEST_DEVICE)
            )
            labels = torch.randint(0, num_classes, (batch_size,))
            pair_indices = (
                torch.randint(0, batch_size, (batch_size,)),
                torch.randint(0, batch_size, (batch_size,)),
            )
            triplet_indices = (
                *pair_indices,
                torch.randint(0, batch_size, (batch_size,)),
            )
            losses = torch.randn(batch_size).type(dtype).to(TEST_DEVICE)

            for indices, reduction_type in [
                (torch.arange(batch_size), "element"),
                (pair_indices, "pos_pair"),
                (pair_indices, "neg_pair"),
                (triplet_indices, "triplet"),
            ]:
                loss_dict = {
                    "loss": {
                        "losses": losses,
                        "indices": indices,
                        "reduction_type": reduction_type,
                    }
                }
                output = reducer(loss_dict, embeddings, labels)
                correct_output = torch.zeros(
                    (), dtype=torch.float64, device=TEST_DEVICE
                )
                for i in range(len(losses)):
                    if reduction_type == "element":
                        batch_idx = indices[i]
                    else:
                        batch_idx = indices[0][i]
                    class_label = labels[batch_idx]
                    correct_output += (
                        losses[i].double()
                        * class_weights.double().to(TEST_DEVICE)[class_label]
                    )
                correct_output /= len(losses)
                helpers.assert_close(output, correct_output, dtype)


@pytest.mark.parametrize("dtype", TEST_DTYPES)
@pytest.mark.parametrize("case", ["ordinary", "large", "small_weight"])
def test_class_weighted_reduction_matches_wide_value_and_gradient(dtype, case):
    if case == "large":
        data, coefficients = [40000.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]
    elif case == "small_weight":
        data, coefficients = [10000.0], [1e-8]
    else:
        data = [17.25, -3.5, 1.125, 7.75]
        coefficients = [1.123, 0.517, 1.99, 0.123]
    losses = torch.tensor(data, dtype=dtype, device=TEST_DEVICE, requires_grad=True)
    weights = torch.tensor(coefficients, dtype=torch.float64, device=TEST_DEVICE)
    labels = torch.arange(len(data), device=TEST_DEVICE)
    anchors = torch.arange(len(data), device=TEST_DEVICE).flip(0)
    reducer = ClassWeightedReducer(weights)
    reference_losses = losses.detach().double().requires_grad_()
    expected = sum(
        reference_losses[i] * weights[labels[anchors[i]]] for i in range(len(data))
    ) / len(data)

    actual = reducer(
        {"loss": {"losses": losses, "indices": anchors}},
        torch.zeros(len(data), 2, dtype=dtype, device=TEST_DEVICE),
        labels,
    )

    torch.testing.assert_close(
        actual,
        expected.detach().to(dtype),
        rtol=1e-6 if dtype == torch.float32 else 0,
        atol=0,
    )
    actual.backward()
    expected.backward()
    torch.testing.assert_close(
        losses.grad,
        reference_losses.grad.to(dtype),
        rtol=1e-6 if dtype == torch.float32 else 0,
        atol=0,
    )
    assert reducer.weights.dtype == torch.float64
    torch.testing.assert_close(reducer.weights, weights)
