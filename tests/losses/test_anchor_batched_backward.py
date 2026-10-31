import unittest

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
    "GeneralizedLiftedStructureLoss": lambda: losses.GeneralizedLiftedStructureLoss(),
    "MultiSimilarityLoss": lambda: losses.MultiSimilarityLoss(),
    "NTXentLoss": lambda: losses.NTXentLoss(),
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

    def test_loss_with_no_nonzero_elements(self):
        embeddings = torch.tensor(
            [[1.0, 0], [1, 0], [-1, 0], [-1, 0]],
            device=TEST_DEVICE,
            requires_grad=True,
        )
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        value = anchor_batched_backward(
            losses.ContrastiveLoss(pos_margin=0, neg_margin=1), embeddings, labels, 2
        )
        self.assertEqual(value.item(), 0.0)
        self.assertIsNone(embeddings.grad)

    def test_inputs_are_not_changed(self):
        embeddings, labels = batch()
        before = embeddings.detach().clone()
        anchor_batched_backward(losses.NTXentLoss(), embeddings, labels, 5)
        self.assertTrue(torch.equal(embeddings.detach(), before))

    def test_rejects_objectives_that_do_not_split_by_anchor(self):
        embeddings, labels = batch()
        cases = [
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
