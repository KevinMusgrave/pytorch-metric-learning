import unittest
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.distributed import DistributedLossWrapper
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning.types import PairIndices

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
    logits = (
        F.normalize(embeddings.double(), dim=1)
        @ F.normalize(weights.double(), dim=1).t()
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
                loss = super().forward(*args, **kwargs)
                assert isinstance(loss, torch.Tensor)
                return 2 * loss

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

    def test_wrappers_reject_foreign_modules_and_losses_without_references(self):
        class ContrastiveLoss(torch.nn.Module):
            pass

        embeddings, labels = self.fixture(torch.float32)
        foreign: Any = ContrastiveLoss()
        cases: tuple[tuple[Any, dict[str, Any], tuple[Tensor, Tensor]], ...] = (
            (
                losses.SelfSupervisedLoss,
                {"symmetric": False},
                (embeddings[:4], embeddings[4:]),
            ),
            (losses.CrossBatchMemory, {"embedding_size": 3}, (embeddings, labels)),
        )
        for wrapper, kwargs, args in cases:
            with self.subTest(wrapper=wrapper):
                with self.assertRaises(TypeError):
                    wrapper(foreign, **kwargs)
                wrapped = wrapper(losses.ProxyNCALoss(3, 3), **kwargs).to(TEST_DEVICE)
                with self.assertRaisesRegex(ValueError, "ref_emb"):
                    wrapped(*args)

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
                weights = loss_func.weight
                expected = (
                    normalized_softmax_reference(embeddings, labels, weights, 0.5)
                    + 0.2 * weights.norm(dim=1).mean()
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
            embedding_regularizer=LpRegularizer(),
            embedding_reg_weight=0.5,
        )
        output = loss_func.loss_terms(embeddings, labels)
        self.assertEqual(set(output), {"loss", "embedding_reg_loss"})
        self.assertIsNone(output["embedding_reg_loss"]["indices"])
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
                self.assertTrue(issubclass(loss_class, losses.BaseProxyLoss))
                loss_func = loss_class(
                    num_classes=3,
                    embedding_size=4,
                    weight_regularizer=LpRegularizer(),
                    weight_reg_weight=0.5,
                )
                embeddings = torch.randn(3, 4)
                terms = loss_func.loss_terms(embeddings, torch.tensor([0, 1, 2]))
                torch.testing.assert_close(
                    terms["weight_reg_loss"]["losses"],
                    0.5 * loss_func.weight.norm(dim=1).mean(),
                )

    def test_forward_does_not_change_parameter_placement(self):
        loss_func = losses.NormalizedSoftmaxLoss(3, 4)
        optimizer = torch.optim.Adam(loss_func.parameters(), lr=0.1)
        labels = torch.tensor([0, 1, 2])
        loss_func(torch.randn(3, 4), labels).backward()
        optimizer.step()
        with self.assertRaisesRegex(ValueError, "float32.*float64|float64.*float32"):
            loss_func(torch.randn(3, 4, dtype=torch.float64), labels)
        self.assertEqual(loss_func.weight.dtype, torch.float32)
        loss_func(torch.randn(3, 4), labels).backward()
        optimizer.step()


def assert_close_to_reference(actual, expected, dtype):
    tolerance = (1e-2, 1e-2) if dtype == torch.float16 else (1e-5, 1e-6)
    torch.testing.assert_close(
        actual.double(), expected.double(), rtol=tolerance[0], atol=tolerance[1]
    )


def nonzero_mean(values):
    kept = values[values > 0]
    return kept.sum() / max(kept.numel(), 1)


def label_masks(labels, ref_labels, self_offset=None):
    positive = labels[:, None] == ref_labels[None, :]
    if self_offset is not None:
        rows = torch.arange(len(labels), device=labels.device)
        positive[rows, rows + self_offset] = False
    return positive, labels[:, None] != ref_labels[None, :]


def index_masks(pairs, shape):
    positive = torch.zeros(shape, dtype=torch.bool, device=pairs[0].device)
    negative = torch.zeros_like(positive)
    positive[pairs[0], pairs[1]] = True
    negative[pairs[2], pairs[3]] = True
    return positive, negative


def combined_reference(query, reference, positive, negative):
    distances = torch.linalg.vector_norm(
        query.double()[:, None] - reference.double()[None, :], dim=-1
    )
    contrastive = nonzero_mean(distances[positive]) + nonzero_mean(
        F.relu(1 - distances[negative])
    )
    triplets = positive[:, :, None] & negative[:, None, :]
    violations = F.relu(distances[:, :, None] - distances[:, None, :] + 0.3)
    return contrastive + 0.5 * nonzero_mean(violations[triplets])


def make_multiple_losses():
    distance = LpDistance(normalize_embeddings=False)
    return losses.MultipleLosses(
        {
            "contrastive": losses.ContrastiveLoss(distance=distance),
            "triplet": losses.TripletMarginLoss(margin=0.3, distance=distance),
        },
        weights={"contrastive": 1.0, "triplet": 0.5},
    )


class TestMultipleLossesComposition(unittest.TestCase):
    def fixture(self, dtype):
        torch.manual_seed(11)
        embeddings = torch.randn(8, 3, device=TEST_DEVICE, dtype=dtype)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1], device=TEST_DEVICE)
        return embeddings, labels

    def assert_matches(self, actual, expected, leaves, dtype):
        assert_close_to_reference(actual, expected, dtype)
        actual_grads = torch.autograd.grad(actual, leaves)
        expected_grads = torch.autograd.grad(expected, leaves)
        for a, e in zip(actual_grads, expected_grads, strict=True):
            self.assertGreater(e.abs().sum().item(), 0)
            assert_close_to_reference(a, e, dtype)

    def test_forward_accepts_references_and_omitted_labels(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                inputs, labels = self.fixture(dtype)
                loss_func = make_multiple_losses()
                x = inputs.clone().requires_grad_()
                expected = combined_reference(x, x, *label_masks(labels, labels, 0))
                self.assert_matches(loss_func(x, labels), expected, [x], dtype)

                query = inputs[:4].clone().requires_grad_()
                reference = (inputs[4:] + 0.2).requires_grad_()
                ref_labels = torch.tensor([0, 1, 1, 0], device=TEST_DEVICE)
                expected = combined_reference(
                    query, reference, *label_masks(labels[:4], ref_labels)
                )
                actual = loss_func(query, labels[:4], None, reference, ref_labels)
                self.assert_matches(actual, expected, [query, reference], dtype)

                pairs = PairIndices(
                    torch.tensor([0, 2, 5], device=TEST_DEVICE),
                    torch.tensor([1, 3, 4], device=TEST_DEVICE),
                    torch.tensor([0, 0, 2, 5], device=TEST_DEVICE),
                    torch.tensor([2, 4, 7, 6], device=TEST_DEVICE),
                )
                x = inputs.clone().requires_grad_()
                expected = combined_reference(x, x, *index_masks(pairs, (8, 8)))
                self.assert_matches(loss_func(x, None, pairs), expected, [x], dtype)

    def test_cross_batch_memory_wraps_multiple_losses(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                inputs, labels = self.fixture(dtype)
                memory = losses.CrossBatchMemory(
                    make_multiple_losses(), embedding_size=3, memory_size=16
                ).to(TEST_DEVICE, dtype)
                history = []
                for step in (slice(0, 4), slice(4, 8)):
                    query = inputs[step].clone().requires_grad_()
                    history.append(query.detach())
                    queued = torch.cat(history)
                    queued_labels = labels[: step.stop]
                    expected = combined_reference(
                        query,
                        queued,
                        *label_masks(labels[step], queued_labels, step.start),
                    )
                    actual = memory(query, labels[step])
                    self.assert_matches(actual, expected, [query], dtype)
                self.assertEqual(memory.queue_idx, 8)

    def test_self_supervised_loss_wraps_multiple_losses(self):
        for dtype in TEST_DTYPES:
            for symmetric in (True, False):
                with self.subTest(dtype=dtype, symmetric=symmetric):
                    inputs, _ = self.fixture(dtype)
                    view_a = inputs[:4].clone().requires_grad_()
                    view_b = (inputs[:4] + 0.3 * inputs[4:]).requires_grad_()
                    ids = torch.arange(4, device=TEST_DEVICE)
                    if symmetric:
                        both = torch.cat([view_a, view_b])
                        both_ids = torch.cat([ids, ids])
                        masks = label_masks(both_ids, both_ids, 0)
                        expected = combined_reference(both, both, *masks)
                    else:
                        masks = label_masks(ids, ids)
                        expected = combined_reference(view_a, view_b, *masks)
                    wrapper = losses.SelfSupervisedLoss(
                        make_multiple_losses(), symmetric=symmetric
                    )
                    actual = wrapper(view_a, view_b)
                    self.assert_matches(actual, expected, [view_a, view_b], dtype)

    def test_distributed_wrapper_without_process_group_wraps_multiple_losses(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                inputs, labels = self.fixture(dtype)
                wrapper = DistributedLossWrapper(make_multiple_losses())
                x = inputs.clone().requires_grad_()
                expected = combined_reference(x, x, *label_masks(labels, labels, 0))
                self.assert_matches(wrapper(x, labels), expected, [x], dtype)
                query = inputs[:4].clone().requires_grad_()
                reference = (inputs[4:] - 0.1).requires_grad_()
                expected = combined_reference(
                    query, reference, *label_masks(labels[:4], labels[4:])
                )
                actual = wrapper(
                    query, labels[:4], ref_emb=reference, ref_labels=labels[4:]
                )
                self.assert_matches(actual, expected, [query, reference], dtype)

    def test_multiple_losses_accepts_only_metric_losses(self):
        memory = losses.CrossBatchMemory(losses.ContrastiveLoss(), embedding_size=3)
        invalid_losses: tuple[Any, ...] = (
            memory,
            losses.VICRegLoss(),
            torch.nn.Identity(),
        )
        for invalid in invalid_losses:
            with (
                self.subTest(invalid=type(invalid).__name__),
                self.assertRaisesRegex(TypeError, "BaseMetricLossFunction"),
            ):
                losses.MultipleLosses({"invalid": invalid})
        nested = losses.MultipleLosses({"inner": make_multiple_losses()})
        inputs, labels = self.fixture(torch.float64)
        torch.testing.assert_close(
            nested(inputs, labels), make_multiple_losses()(inputs, labels)
        )


class TestLossTermNames(unittest.TestCase):
    def test_objective_terms_cannot_reuse_regularization_names(self):
        class RenamedTerms(losses.ContrastiveLoss):
            def pair_loss(self, mat, pairs):
                terms = super().pair_loss(mat, pairs)
                return {"embedding_reg_loss": terms["pos_loss"], **terms}

        class RenamedProxyTerms(losses.NormalizedSoftmaxLoss):
            def compute_loss(
                self, embeddings, labels, indices_tuple, ref_emb, ref_labels
            ):
                terms = super().compute_loss(
                    embeddings, labels, indices_tuple, ref_emb, ref_labels
                )
                return {"weight_reg_loss": terms["loss"]}

        embeddings = torch.randn(4, 3, device=TEST_DEVICE)
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        cases = (
            (RenamedTerms(embedding_regularizer=LpRegularizer()), "embedding_reg_loss"),
            (
                RenamedProxyTerms(2, 3, weight_regularizer=LpRegularizer()).to(
                    TEST_DEVICE
                ),
                "weight_reg_loss",
            ),
        )
        for loss_func, name in cases:
            with self.subTest(name=name):
                for call in (loss_func, loss_func.loss_terms):
                    with self.assertRaisesRegex(ValueError, name):
                        call(embeddings, labels)
        unregularized = RenamedTerms()
        terms = unregularized.loss_terms(embeddings, labels)
        self.assertEqual(set(terms), {"embedding_reg_loss", "pos_loss", "neg_loss"})


class TestMinedWeightSelection(unittest.TestCase):
    def test_empty_indices_select_nothing_and_none_selects_everything(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                embeddings = torch.randn(
                    6, 3, device=TEST_DEVICE, dtype=dtype, requires_grad=True
                )
                labels = torch.tensor([0, 1, 2, 0, 1, 2], device=TEST_DEVICE)
                loss_func = losses.NormalizedSoftmaxLoss(3, 3, temperature=0.5).to(
                    TEST_DEVICE, dtype
                )
                weight = loss_func.weight
                logits = (
                    F.normalize(embeddings.double(), dim=1)
                    @ F.normalize(weight.double(), dim=1).t()
                    / 0.5
                )
                per_sample = F.cross_entropy(logits, labels, reduction="none")
                unmined = loss_func(embeddings, labels)
                expected = per_sample.mean()
                torch.testing.assert_close(
                    unmined.double(), expected, **tolerances(dtype)
                )

                empty = torch.empty(0, dtype=torch.long, device=TEST_DEVICE)
                selected_nothing = loss_func(embeddings, labels, (empty,) * 4)
                gradients = torch.autograd.grad(selected_nothing, [embeddings, weight])
                self.assertEqual(selected_nothing.item(), 0)
                for gradient in gradients:
                    self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))

                anchors = torch.tensor([0, 0, 3], device=TEST_DEVICE)
                others = torch.tensor([3, 4, 4], device=TEST_DEVICE)
                counts = torch.zeros(6, dtype=torch.float64, device=TEST_DEVICE)
                for index in (0, 0, 3, 3, 4, 4):
                    counts[index] += 1
                expected = (per_sample * counts / 2).mean()
                selected = loss_func(embeddings, labels, (anchors, others) * 2)
                torch.testing.assert_close(
                    selected.double(), expected, **tolerances(dtype)
                )
                for actual_grad, expected_grad in zip(
                    torch.autograd.grad(selected, [embeddings, weight]),
                    torch.autograd.grad(expected, [embeddings, weight]),
                    strict=True,
                ):
                    torch.testing.assert_close(
                        actual_grad.double(),
                        expected_grad.double(),
                        **tolerances(dtype),
                    )


if __name__ == "__main__":
    unittest.main()
