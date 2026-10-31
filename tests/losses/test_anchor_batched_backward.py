import copy
import unittest
from typing import cast

import torch

from pytorch_metric_learning import distances, losses, reducers, regularizers
from pytorch_metric_learning.losses import anchor_batched_backward

from .. import TEST_DEVICE, TEST_DTYPES

SUPPORTED = {
    "CircleLoss": lambda: losses.CircleLoss(),
    "ContrastiveLoss": lambda: losses.ContrastiveLoss(),
    "ContrastiveLoss-mean": lambda: losses.ContrastiveLoss(
        reducer=reducers.MeanReducer()
    ),
    "DecoupledContrastiveLoss": lambda: losses.DecoupledContrastiveLoss(),
    "GeneralizedLiftedStructureLoss": lambda: losses.GeneralizedLiftedStructureLoss(),
    "MultiSimilarityLoss": lambda: losses.MultiSimilarityLoss(),
    "NTXentLoss": lambda: losses.NTXentLoss(),
    "SelectivelyContrastiveTripletLoss": lambda: (
        losses.SelectivelyContrastiveTripletLoss()
    ),
    "SignalToNoiseRatioContrastiveLoss": lambda: (
        losses.SignalToNoiseRatioContrastiveLoss()
    ),
    "SupConLoss": lambda: losses.SupConLoss(),
    "TripletMarginLoss": lambda: losses.TripletMarginLoss(),
    "TripletMarginLoss-smooth": lambda: losses.TripletMarginLoss(smooth_loss=True),
    "TripletMarginLoss-swap": lambda: losses.TripletMarginLoss(swap=True),
    "TupletMarginLoss": lambda: losses.TupletMarginLoss(),
}


def batch(dtype=torch.float64, size=18):
    generator = torch.Generator().manual_seed(2)
    embeddings = torch.randn(size, 8, generator=generator, dtype=dtype)
    return embeddings.to(TEST_DEVICE).requires_grad_(), (torch.arange(size) % 4).to(
        TEST_DEVICE
    )


def references(dtype=torch.float64):
    generator = torch.Generator().manual_seed(7)
    ref_emb = torch.randn(11, 8, generator=generator, dtype=dtype)
    return ref_emb.to(TEST_DEVICE).requires_grad_(), (torch.arange(11) % 5).to(
        TEST_DEVICE
    )


class LearnedCosine(distances.CosineSimilarity):
    scale: torch.nn.Parameter

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def compute_mat(self, query_emb, ref_emb):
        return self.scale * super().compute_mat(query_emb, ref_emb)


def scale_of(loss_func):
    return cast(LearnedCosine, loss_func.distance).scale


class TestAnchorBatchedBackward(unittest.TestCase):
    def assert_matches(self, actual, expected, dtype):
        tolerance = {torch.float16: 2e-3, torch.float32: 1e-5}.get(dtype, 1e-11)
        scale = expected.abs().max().item()
        torch.testing.assert_close(
            actual.double(), expected.double(), rtol=0, atol=tolerance * scale
        )

    def test_matches_full_batch_value_and_gradient(self):
        for name, factory in SUPPORTED.items():
            for dtype in TEST_DTYPES:
                for batch_size in (1, 5, 18, 100):
                    with self.subTest(loss=name, dtype=dtype, batch_size=batch_size):
                        embeddings, labels = batch(dtype)
                        wide = embeddings.detach().double().requires_grad_()
                        loss_func = factory()
                        expected = loss_func(wide, labels)
                        expected.backward()
                        value = anchor_batched_backward(
                            loss_func, embeddings, labels, batch_size
                        )
                        assert embeddings.grad is not None and wide.grad is not None
                        self.assertEqual(value.dtype, dtype)
                        self.assertEqual(embeddings.grad.dtype, dtype)
                        self.assertFalse(value.requires_grad)
                        self.assert_matches(value, expected.detach(), dtype)
                        self.assert_matches(embeddings.grad, wide.grad, dtype)
                        self.assertTrue(embeddings.grad.any())

    def test_matches_full_batch_with_reference_set(self):
        for name, factory in SUPPORTED.items():
            for dtype in TEST_DTYPES:
                with self.subTest(loss=name, dtype=dtype):
                    embeddings, labels = batch(dtype)
                    ref_emb, ref_labels = references(dtype)
                    wide = embeddings.detach().double().requires_grad_()
                    wide_ref = ref_emb.detach().double().requires_grad_()
                    loss_func = factory()
                    expected = loss_func(
                        wide, labels, ref_emb=wide_ref, ref_labels=ref_labels
                    )
                    expected.backward()
                    value = anchor_batched_backward(
                        loss_func,
                        embeddings,
                        labels,
                        4,
                        ref_emb=ref_emb,
                        ref_labels=ref_labels,
                    )
                    assert embeddings.grad is not None and wide.grad is not None
                    assert ref_emb.grad is not None and wide_ref.grad is not None
                    self.assert_matches(value, expected.detach(), dtype)
                    self.assert_matches(embeddings.grad, wide.grad, dtype)
                    self.assert_matches(ref_emb.grad, wide_ref.grad, dtype)
                    self.assertTrue(ref_emb.grad.any())

    def test_detached_reference_receives_no_gradient(self):
        embeddings, labels = batch()
        ref_emb, ref_labels = references()
        ref_emb = ref_emb.detach()
        loss_func = losses.ContrastiveLoss()
        expected = embeddings.detach().clone().requires_grad_()
        loss_func(expected, labels, ref_emb=ref_emb, ref_labels=ref_labels).backward()
        anchor_batched_backward(
            loss_func, embeddings, labels, 3, ref_emb=ref_emb, ref_labels=ref_labels
        )
        assert embeddings.grad is not None and expected.grad is not None
        self.assert_matches(embeddings.grad, expected.grad, torch.float64)
        self.assertIsNone(ref_emb.grad)

    def test_gradient_reaches_the_encoder(self):
        features, labels = batch()
        features = features.detach()
        full = torch.nn.Linear(8, 4, dtype=torch.float64).to(TEST_DEVICE)
        batched = torch.nn.Linear(8, 4, dtype=torch.float64).to(TEST_DEVICE)
        batched.load_state_dict(full.state_dict())
        loss_func = losses.NTXentLoss()
        expected = loss_func(full(features), labels)
        expected.backward()
        value = anchor_batched_backward(loss_func, batched(features), labels, 4)
        self.assert_matches(value, expected.detach(), torch.float64)
        assert batched.weight.grad is not None and full.weight.grad is not None
        self.assert_matches(batched.weight.grad, full.weight.grad, torch.float64)

    def test_gradients_accumulate_like_backward(self):
        embeddings, labels = batch()
        loss_func = losses.ContrastiveLoss()
        anchor_batched_backward(loss_func, embeddings, labels, 5)
        assert embeddings.grad is not None
        once = embeddings.grad.clone()
        anchor_batched_backward(loss_func, embeddings, labels, 5)
        torch.testing.assert_close(embeddings.grad, 2 * once)

    def test_no_chunk_exceeds_batch_size(self):
        shapes = []

        class Recording(distances.LpDistance):
            def forward(self, query_emb, ref_emb=None):
                shapes.append(len(query_emb))
                return super().forward(query_emb, ref_emb)

        embeddings, labels = batch()
        anchor_batched_backward(
            losses.ContrastiveLoss(distance=Recording()), embeddings, labels, 5
        )
        self.assertEqual(shapes, [5, 5, 5, 3])

    def test_labels_without_tuples_return_a_zero(self):
        embeddings = torch.randn(6, 4, device=TEST_DEVICE, requires_grad=True)
        labels = torch.arange(6, device=TEST_DEVICE)
        for loss_func in (
            losses.TripletMarginLoss(),
            losses.SupConLoss(),
        ):
            with self.subTest(loss=type(loss_func).__name__):
                full = loss_func(embeddings, labels)
                value = anchor_batched_backward(loss_func, embeddings, labels, 2)
                self.assertEqual(value.item(), full.item())

    def assert_same_gradient(self, actual, expected):
        self.assertEqual(actual is None, expected is None)
        if expected is not None:
            self.assert_matches(actual, expected, torch.float64)

    def test_learnable_distance_trains_like_a_full_batch(self):
        for batch_size in (1, 3, 18):
            for use_reference in (False, True):
                with self.subTest(batch_size=batch_size, reference=use_reference):
                    embeddings, labels = batch()
                    full_x = embeddings.detach().clone().requires_grad_()
                    options = {}
                    if use_reference:
                        ref_emb, ref_labels = references()
                        options = {"ref_emb": ref_emb, "ref_labels": ref_labels}
                    full = losses.NTXentLoss(distance=LearnedCosine()).to(TEST_DEVICE)
                    batched = copy.deepcopy(full)
                    full_optimizer = torch.optim.AdamW(full.parameters(), lr=0.1)
                    optimizer = torch.optim.AdamW(batched.parameters(), lr=0.1)
                    expected = full(full_x, labels, **options)
                    expected.backward()
                    value = anchor_batched_backward(
                        batched, embeddings, labels, batch_size, **options
                    )
                    self.assert_matches(value, expected.detach(), torch.float64)
                    self.assertTrue(scale_of(full).grad.abs() > 0)
                    self.assert_same_gradient(
                        scale_of(batched).grad, scale_of(full).grad
                    )
                    full_optimizer.step()
                    optimizer.step()
                    self.assert_matches(
                        scale_of(batched).detach(),
                        scale_of(full).detach(),
                        torch.float64,
                    )

    def test_low_precision_parameter_gradients_stay_finite(self):
        cases = [
            (torch.float16, 128, 64, 16),
            (torch.float16, 256, 64, 256),
            (torch.bfloat16, 128, 64, 16),
            (torch.float32, 128, 64, 16),
        ]
        for dtype, size, group, chunk in cases:
            with self.subTest(dtype=dtype, size=size, chunk=chunk):
                torch.manual_seed(3)
                embeddings = torch.randn(
                    size, 8, dtype=dtype, device=TEST_DEVICE, requires_grad=True
                )
                labels = (torch.arange(size) // group).to(TEST_DEVICE)
                reference = embeddings.detach().double().requires_grad_()
                full = losses.NTXentLoss(temperature=0.07, distance=LearnedCosine())
                full = full.to(TEST_DEVICE)
                full(reference, labels).backward()
                batched = losses.NTXentLoss(
                    temperature=0.07, distance=LearnedCosine()
                ).to(TEST_DEVICE, dtype)
                anchor_batched_backward(batched, embeddings, labels, chunk)
                gradient = scale_of(batched).grad
                assert gradient is not None
                expected = scale_of(full).grad
                assert expected is not None
                self.assertEqual(gradient.dtype, dtype)
                self.assertEqual(scale_of(batched).dtype, dtype)
                self.assertTrue(gradient.isfinite())
                torch.testing.assert_close(
                    gradient.double(), expected, rtol=4e-3, atol=0
                )

    def test_parameter_hook_sees_the_final_gradient_once(self):
        embeddings, labels = batch()
        full_x = embeddings.detach().clone().requires_grad_()
        full = losses.NTXentLoss(distance=LearnedCosine()).to(TEST_DEVICE)
        batched = copy.deepcopy(full)
        full(full_x, labels).backward()
        expected = scale_of(full).grad
        assert expected is not None
        seen = []

        def halve(gradient):
            seen.append(gradient.clone())
            return gradient * 0.5

        scale_of(batched).register_hook(halve)
        anchor_batched_backward(batched, embeddings, labels, 3)
        self.assertEqual(len(seen), 1)
        self.assert_matches(seen[0], expected, torch.float64)
        gradient = scale_of(batched).grad
        assert gradient is not None
        self.assert_matches(gradient, expected * 0.5, torch.float64)

    def test_input_hooks_see_the_final_gradient_once(self):
        embeddings, labels = batch()
        ref_emb, ref_labels = references()
        full_x = embeddings.detach().clone().requires_grad_()
        full_ref = ref_emb.detach().clone().requires_grad_()
        loss_func = losses.ContrastiveLoss()
        loss_func(full_x, labels, ref_emb=full_ref, ref_labels=ref_labels).backward()
        assert full_x.grad is not None and full_ref.grad is not None
        seen = {"embeddings": [], "reference": []}
        embeddings.register_hook(lambda g: seen["embeddings"].append(g.clone()))
        ref_emb.register_hook(lambda g: seen["reference"].append(g.clone()))
        anchor_batched_backward(
            loss_func, embeddings, labels, 3, ref_emb=ref_emb, ref_labels=ref_labels
        )
        self.assertEqual([len(x) for x in seen.values()], [1, 1])
        self.assert_matches(seen["embeddings"][0], full_x.grad, torch.float64)
        self.assert_matches(seen["reference"][0], full_ref.grad, torch.float64)

    def test_parameter_gradients_accumulate_like_backward(self):
        embeddings, labels = batch()
        loss_func = losses.NTXentLoss(distance=LearnedCosine()).to(TEST_DEVICE)
        anchor_batched_backward(loss_func, embeddings, labels, 5)
        once = scale_of(loss_func).grad
        assert once is not None
        once = once.clone()
        anchor_batched_backward(loss_func, embeddings, labels, 5)
        gradient = scale_of(loss_func).grad
        assert gradient is not None
        torch.testing.assert_close(gradient, 2 * once)

    def test_parameters_are_restored_after_each_call(self):
        embeddings, labels = batch()
        loss_func = losses.NTXentLoss(distance=LearnedCosine()).to(TEST_DEVICE)
        parameter = scale_of(loss_func)
        anchor_batched_backward(loss_func, embeddings, labels, 5)
        self.assertIs(scale_of(loss_func), parameter)
        self.assertEqual(
            dict(loss_func.named_parameters())["distance.scale"].numel(), 1
        )

        class Failing(LearnedCosine):
            def compute_mat(self, query_emb, ref_emb):
                raise RuntimeError("chunk failed")

        failing = losses.NTXentLoss(distance=Failing()).to(TEST_DEVICE)
        parameter = scale_of(failing)
        with self.assertRaisesRegex(RuntimeError, "chunk failed"):
            anchor_batched_backward(failing, embeddings, labels, 5)
        self.assertIs(scale_of(failing), parameter)
        self.assertIsNone(parameter.grad)

    def test_parameters_full_backward_skips_keep_no_gradient(self):
        embeddings, labels = batch()
        full_x = embeddings.detach().clone().requires_grad_()
        full = losses.NTXentLoss(distance=LearnedCosine()).to(TEST_DEVICE)
        unused = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        frozen = torch.nn.Parameter(
            torch.ones(1, dtype=torch.float64), requires_grad=False
        )
        full.distance.register_parameter("unused", unused)
        full.distance.register_parameter("frozen", frozen)
        batched = copy.deepcopy(full)
        full(full_x, labels).backward()
        anchor_batched_backward(batched, embeddings, labels, 5)
        parameters = dict(batched.named_parameters())
        self.assertIsNone(unused.grad)
        self.assertIsNone(parameters["distance.unused"].grad)
        self.assertIsNone(parameters["distance.frozen"].grad)

    def test_satisfied_batch_matches_full_backward_state(self):
        separated = torch.tensor(
            [[1.0, 0], [1, 0], [-1, 0], [-1, 0]],
            dtype=torch.float64,
            device=TEST_DEVICE,
        )
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        cases = {
            "all-zero-terms": (
                losses.ContrastiveLoss(pos_margin=0, neg_margin=1),
                labels,
            ),
            "no-tuples": (losses.TripletMarginLoss(), torch.arange(4).to(TEST_DEVICE)),
        }
        for name, (loss_func, targets) in cases.items():
            with self.subTest(case=name):
                full_model = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
                torch.nn.init.eye_(full_model.weight)
                full_model = full_model.to(TEST_DEVICE)
                model = copy.deepcopy(full_model)
                full_optimizer = torch.optim.AdamW(
                    full_model.parameters(), lr=0.1, weight_decay=0.1
                )
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=0.1, weight_decay=0.1
                )
                expected = loss_func(full_model(separated), targets)
                expected.backward()
                value = anchor_batched_backward(loss_func, model(separated), targets, 2)
                self.assertEqual(value.item(), expected.item())
                assert model.weight.grad is not None
                assert full_model.weight.grad is not None
                self.assertTrue(torch.equal(model.weight.grad, full_model.weight.grad))
                full_optimizer.step()
                optimizer.step()
                self.assertTrue(torch.equal(model.weight, full_model.weight))
                self.assertFalse(
                    torch.equal(model.weight, torch.eye(2, device=TEST_DEVICE).double())
                )

    def test_zero_terms_reach_learnable_distances(self):
        separated = torch.tensor(
            [[1.0, 0], [1, 0], [-1, 0], [-1, 0]],
            dtype=torch.float64,
            device=TEST_DEVICE,
        )
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        full = losses.ContrastiveLoss(
            pos_margin=0.5, neg_margin=0.5, distance=LearnedCosine()
        ).to(TEST_DEVICE)
        batched = copy.deepcopy(full)
        full_x = separated.clone().requires_grad_()
        x = separated.clone().requires_grad_()
        full(full_x, labels).backward()
        anchor_batched_backward(batched, x, labels, 2)
        assert full_x.grad is not None and x.grad is not None
        self.assertEqual(full_x.grad.abs().sum().item(), 0)
        self.assertEqual(x.grad.abs().sum().item(), 0)
        self.assert_same_gradient(scale_of(batched).grad, scale_of(full).grad)

    def test_empty_term_leaves_learnable_distance_untouched(self):
        embeddings, _ = batch()
        labels = torch.arange(len(embeddings), device=TEST_DEVICE)
        full_x = embeddings.detach().clone().requires_grad_()
        full = losses.NTXentLoss(distance=LearnedCosine()).to(TEST_DEVICE)
        batched = copy.deepcopy(full)
        full(full_x, labels).backward()
        anchor_batched_backward(batched, embeddings, labels, 5)
        self.assertIsNone(scale_of(full).grad)
        self.assertIsNone(scale_of(batched).grad)
        self.assert_same_gradient(embeddings.grad, full_x.grad)

    def test_reference_gradient_state_matches_without_tuples(self):
        embeddings = torch.randn(
            4, 2, dtype=torch.float64, device=TEST_DEVICE, requires_grad=True
        )
        ref_emb = torch.randn(
            3, 2, dtype=torch.float64, device=TEST_DEVICE, requires_grad=True
        )
        labels = torch.arange(4, device=TEST_DEVICE)
        ref_labels = torch.arange(10, 13, device=TEST_DEVICE)
        loss_func = losses.TripletMarginLoss()
        full_x = embeddings.detach().clone().requires_grad_()
        full_ref = ref_emb.detach().clone().requires_grad_()
        loss_func(full_x, labels, ref_emb=full_ref, ref_labels=ref_labels).backward()
        anchor_batched_backward(
            loss_func, embeddings, labels, 2, ref_emb=ref_emb, ref_labels=ref_labels
        )
        self.assert_same_gradient(embeddings.grad, full_x.grad)
        self.assert_same_gradient(ref_emb.grad, full_ref.grad)

    def test_inputs_are_not_changed(self):
        embeddings, labels = batch()
        before = embeddings.detach().clone()
        anchor_batched_backward(losses.NTXentLoss(), embeddings, labels, 5)
        self.assertTrue(torch.equal(embeddings.detach(), before))

    def test_rejects_objectives_that_do_not_split_by_anchor(self):
        embeddings, labels = batch()
        cases = [
            (losses.DecoupledContrastiveLoss(sigma=0.5), "cannot be batched"),
            (losses.IntraPairVarianceLoss(), "cannot be batched"),
            (losses.MarginLoss(), "cannot be batched"),
            (losses.NCALoss(), "cannot be batched"),
            (losses.LiftedStructureLoss(), "cannot be batched"),
            (losses.ContrastiveLoss(reducer=reducers.SumReducer()), "only MeanReducer"),
            (
                losses.ContrastiveLoss(reducer=reducers.ThresholdReducer(low=0.1)),
                "only MeanReducer",
            ),
            (
                losses.ContrastiveLoss(
                    embedding_regularizer=regularizers.LpRegularizer()
                ),
                "regularizers",
            ),
            (losses.TripletMarginLoss(triplets_per_anchor=2), "triplets_per_anchor"),
        ]
        for loss_func, message in cases:
            with (
                self.subTest(message=message, loss=type(loss_func).__name__),
                self.assertRaisesRegex(ValueError, message),
            ):
                anchor_batched_backward(loss_func, embeddings, labels, 4)

    def test_every_other_exported_loss_is_rejected(self):
        supported = {name.split("-")[0] for name in SUPPORTED}
        for name, value in vars(losses).items():
            if (
                isinstance(value, type)
                and issubclass(value, losses.BaseMetricLossFunction)
                and name not in supported
            ):
                with self.subTest(loss=name):
                    self.assertFalse(value.decomposes_by_anchor)

    def test_invalid_arguments(self):
        embeddings, labels = batch()
        loss_func = losses.ContrastiveLoss()
        for size in (0, -1):
            with (
                self.subTest(size=size),
                self.assertRaisesRegex(ValueError, "batch_size"),
            ):
                anchor_batched_backward(loss_func, embeddings, labels, size)
        with self.assertRaisesRegex(ValueError, "tracking gradients"):
            anchor_batched_backward(loss_func, embeddings.detach(), labels, 4)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "tracking gradients"):
            anchor_batched_backward(loss_func, embeddings, labels, 4)
        with self.assertRaises(ValueError):
            anchor_batched_backward(loss_func, embeddings, labels[:3], 4)
