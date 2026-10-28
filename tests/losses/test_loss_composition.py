import inspect
import unittest

import torch
from torch.nn import functional as F

from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.regularizers import LpRegularizer

from .. import TEST_DEVICE, TEST_DTYPES


def tolerances(dtype):
    if dtype == torch.float16:
        return {"rtol": 1e-2, "atol": 1e-2}
    return {}


def contrastive_reference(embeddings, labels):
    normalized = F.normalize(embeddings.double(), dim=1)
    distances = torch.cdist(normalized, normalized)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    pos = distances[same & ~eye]
    neg = (1 - distances[~same]).clamp_min(0)
    pos_loss = pos[pos > 0].mean()
    neg_loss = neg[neg > 0].mean()
    return (pos_loss + neg_loss).to(embeddings.dtype)


def normalized_softmax_reference(embeddings, labels, weights, temperature):
    logits = F.normalize(embeddings.double(), dim=1) @ F.normalize(
        weights.double(), dim=0
    )
    return F.cross_entropy(logits / temperature, labels).to(embeddings.dtype)


class TestLossComposition(unittest.TestCase):
    def fixture(self, dtype):
        torch.manual_seed(5)
        embeddings = torch.randn(8, 3, device=TEST_DEVICE, dtype=dtype)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1], device=TEST_DEVICE)
        return embeddings.requires_grad_(), labels

    def test_wrappers_preserve_custom_subclass_values_and_gradients(self):
        class ScaledContrastiveLoss(losses.ContrastiveLoss):
            def forward(self, *args, **kwargs):
                return 2 * super().forward(*args, **kwargs)

        for dtype in TEST_DTYPES:
            for mode in ("memory", "symmetric", "asymmetric"):
                with self.subTest(dtype=dtype, mode=mode):
                    inputs, labels = self.fixture(dtype)
                    components = [losses.ContrastiveLoss(), ScaledContrastiveLoss()]
                    if mode == "memory":
                        wrappers = [
                            losses.CrossBatchMemory(component, 3, memory_size=16).to(
                                TEST_DEVICE, dtype
                            )
                            for component in components
                        ]
                        args = (inputs, labels)
                    else:
                        wrappers = [
                            losses.SelfSupervisedLoss(
                                component, symmetric=mode == "symmetric"
                            )
                            for component in components
                        ]
                        args = (inputs[:4], inputs[4:])
                    expected, actual = [wrapper(*args) for wrapper in wrappers]
                    torch.testing.assert_close(
                        actual, 2 * expected, **tolerances(dtype)
                    )
                    torch.testing.assert_close(
                        torch.autograd.grad(actual, inputs)[0],
                        2 * torch.autograd.grad(expected, inputs)[0],
                        **tolerances(dtype),
                    )

    def test_wrappers_reject_unsupported_types_and_proxy_nca_subclasses(self):
        class ContrastiveLoss(torch.nn.Module):
            pass

        class CustomProxyNCA(losses.ProxyNCALoss):
            pass

        for component in (
            ContrastiveLoss(),
            losses.ProxyNCALoss(3, 3),
            CustomProxyNCA(3, 3),
        ):
            for wrapper in (losses.SelfSupervisedLoss, losses.CrossBatchMemory):
                with self.subTest(component=type(component), wrapper=wrapper):
                    kwargs = (
                        {"embedding_size": 3}
                        if wrapper is losses.CrossBatchMemory
                        else {}
                    )
                    with self.assertRaisesRegex(TypeError, "not supported"):
                        wrapper(component, **kwargs)

    def test_embedding_regularization_adds_weighted_penalty(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                embeddings, labels = self.fixture(dtype)
                loss_func = losses.ContrastiveLoss(
                    embedding_regularizer=LpRegularizer(), embedding_reg_weight=0.3
                )
                actual = loss_func(embeddings, labels)
                expected = (
                    contrastive_reference(embeddings, labels)
                    + 0.3 * embeddings.norm(dim=1).mean()
                )
                torch.testing.assert_close(actual, expected, **tolerances(dtype))
                torch.testing.assert_close(
                    torch.autograd.grad(actual, embeddings)[0],
                    torch.autograd.grad(expected, embeddings)[0],
                    **tolerances(dtype),
                )

    def test_weight_regularization_adds_weighted_penalty(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                embeddings, labels = self.fixture(dtype)
                loss_func = losses.NormalizedSoftmaxLoss(
                    3,
                    3,
                    temperature=0.5,
                    weight_regularizer=LpRegularizer(),
                    weight_reg_weight=0.2,
                    embedding_regularizer=LpRegularizer(power=2),
                    embedding_reg_weight=0.1,
                ).to(TEST_DEVICE, dtype)
                actual = loss_func(embeddings, labels)
                weights = loss_func.W
                expected = (
                    normalized_softmax_reference(embeddings, labels, weights, 0.5)
                    + 0.2 * weights.norm(dim=0).mean()
                    + 0.1 * embeddings.square().sum(dim=1).mean()
                )
                torch.testing.assert_close(actual, expected, **tolerances(dtype))
                actual_grads = torch.autograd.grad(actual, [embeddings, weights])
                expected_grads = torch.autograd.grad(expected, [embeddings, weights])
                for a, e in zip(actual_grads, expected_grads, strict=True):
                    torch.testing.assert_close(a, e, **tolerances(dtype))

    def test_reducers_see_only_objective_terms(self):
        embeddings, labels = self.fixture(torch.float64)
        loss_func = losses.ContrastiveLoss(
            reducer=reducers.MultipleReducers(
                {"pos_loss": reducers.MeanReducer()},
                default_reducer=reducers.SumReducer(),
            ),
            embedding_regularizer=LpRegularizer(),
            embedding_reg_weight=0.5,
        )
        normalized = F.normalize(embeddings, dim=1)
        distances = torch.cdist(normalized, normalized)
        same = labels[:, None] == labels[None, :]
        eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        expected = (
            distances[same & ~eye].mean()
            + (1 - distances[~same]).clamp_min(0).sum()
            + 0.5 * embeddings.norm(dim=1).mean()
        )
        torch.testing.assert_close(loss_func(embeddings, labels), expected)

    def test_unreduced_output_includes_regularization_terms(self):
        embeddings, labels = self.fixture(torch.float64)
        loss_func = losses.TripletMarginLoss(
            reducer=reducers.DoNothingReducer(),
            embedding_regularizer=LpRegularizer(),
            embedding_reg_weight=0.5,
        )
        output = loss_func(embeddings, labels)
        self.assertEqual(set(output), {"loss", "embedding_reg_loss"})
        self.assertEqual(
            output["embedding_reg_loss"]["reduction_type"], "already_reduced"
        )
        torch.testing.assert_close(
            output["embedding_reg_loss"]["losses"], 0.5 * embeddings.norm(dim=1).mean()
        )

    def test_parameter_losses_declare_regularization_arguments(self):
        for loss_class in (
            losses.ArcFaceLoss,
            losses.CosFaceLoss,
            losses.LargeMarginSoftmaxLoss,
            losses.NormalizedSoftmaxLoss,
            losses.ProxyAnchorLoss,
            losses.ProxyNCALoss,
            losses.SoftTripleLoss,
            losses.SphereFaceLoss,
            losses.SubCenterArcFaceLoss,
        ):
            with self.subTest(loss=loss_class.__name__):
                parameters = inspect.signature(loss_class).parameters
                for name in ("weight_regularizer", "weight_reg_weight"):
                    self.assertIn(name, parameters)
                    self.assertEqual(
                        parameters[name].kind, inspect.Parameter.KEYWORD_ONLY
                    )

    def test_forward_does_not_change_parameter_placement(self):
        loss_func = losses.NormalizedSoftmaxLoss(3, 4)
        optimizer = torch.optim.Adam(loss_func.parameters(), lr=0.1)
        labels = torch.tensor([0, 1, 2])
        loss_func(torch.randn(3, 4), labels).backward()
        optimizer.step()
        with self.assertRaisesRegex(ValueError, "float32.*float64|float64.*float32"):
            loss_func(torch.randn(3, 4, dtype=torch.float64), labels)
        self.assertEqual(loss_func.W.dtype, torch.float32)
        loss_func(torch.randn(3, 4), labels).backward()
        optimizer.step()


if __name__ == "__main__":
    unittest.main()
