import copy
import gc
import inspect
import unittest
import weakref

import torch

from pytorch_metric_learning import distances, losses, miners
from pytorch_metric_learning.distributed import (
    DistributedLossWrapper,
    DistributedMinerWrapper,
)

from .. import TEST_DEVICE, TEST_DTYPES, helpers

COSINE = distances.CosineSimilarity
PAIR_LOSSES = {
    "CircleLoss": lambda: losses.CircleLoss(),
    "ContrastiveLoss": lambda: losses.ContrastiveLoss(),
    "GeneralizedLiftedStructureLoss": lambda: losses.GeneralizedLiftedStructureLoss(),
    "IntraPairVarianceLoss": lambda: losses.IntraPairVarianceLoss(),
    "MultiSimilarityLoss": lambda: losses.MultiSimilarityLoss(),
    "NTXentLoss": lambda: losses.NTXentLoss(),
    "SignalToNoiseRatioContrastiveLoss": lambda: (
        losses.SignalToNoiseRatioContrastiveLoss()
    ),
    "SupConLoss": lambda: losses.SupConLoss(),
    "TupletMarginLoss": lambda: losses.TupletMarginLoss(),
}
REFERENCE_LOSSES = {
    **PAIR_LOSSES,
    "DynamicSoftMarginLoss": lambda: losses.DynamicSoftMarginLoss(),
    "MarginLoss": lambda: losses.MarginLoss(),
    "NCALoss": lambda: losses.NCALoss(),
    "TripletMarginLoss": lambda: losses.TripletMarginLoss(),
}
BATCH_ONLY_LOSSES = {
    "FastAPLoss": lambda: losses.FastAPLoss(),
    "HistogramLoss": lambda: losses.HistogramLoss(),
    "InstanceLoss": lambda: losses.InstanceLoss(),
    "LiftedStructureLoss": lambda: losses.LiftedStructureLoss(),
    "PNPLoss": lambda: losses.PNPLoss(),
    "RankedListLoss": lambda: losses.RankedListLoss(
        margin=0.4, neg_temperature=10, alpha=1.2
    ),
    "SmoothAPLoss": lambda: losses.SmoothAPLoss(),
}
MINERS = {
    "BatchEasyHardMiner": lambda: miners.BatchEasyHardMiner(),
    "BatchHardMiner": lambda: miners.BatchHardMiner(),
    "DistanceWeightedMiner": lambda: miners.DistanceWeightedMiner(),
    "HDCMiner": lambda: miners.HDCMiner(),
    "MultiSimilarityMiner": lambda: miners.MultiSimilarityMiner(),
    "PairMarginMiner": lambda: miners.PairMarginMiner(),
    "TripletMarginMiner": lambda: miners.TripletMarginMiner(),
    "UniformHistogramMiner": lambda: miners.UniformHistogramMiner(),
}
UNSUPPORTED_LOSSES = {
    "AngularLoss": lambda: losses.AngularLoss(),
    "ArcFaceLoss": lambda: losses.ArcFaceLoss(num_classes=3, embedding_size=8),
    "CosFaceLoss": lambda: losses.CosFaceLoss(num_classes=3, embedding_size=8),
    "DynamicArcFaceLoss": lambda: losses.DynamicArcFaceLoss(
        class_counts=torch.tensor([3, 4, 5]), embedding_size=8
    ),
    "LargeMarginSoftmaxLoss": lambda: losses.LargeMarginSoftmaxLoss(
        num_classes=3, embedding_size=8
    ),
    "ManifoldLoss": lambda: losses.ManifoldLoss(embedding_size=8),
    "NormalizedSoftmaxLoss": lambda: losses.NormalizedSoftmaxLoss(
        num_classes=3, embedding_size=8
    ),
    "NPairsLoss": lambda: losses.NPairsLoss(),
    "P2SGradLoss": lambda: losses.P2SGradLoss(embedding_size=8, num_classes=3),
    "ProxyAnchorLoss": lambda: losses.ProxyAnchorLoss(num_classes=3, embedding_size=8),
    "ProxyNCALoss": lambda: losses.ProxyNCALoss(num_classes=3, embedding_size=8),
    "SoftTripleLoss": lambda: losses.SoftTripleLoss(num_classes=3, embedding_size=8),
    "SphereFaceLoss": lambda: losses.SphereFaceLoss(num_classes=3, embedding_size=8),
    "SubCenterArcFaceLoss": lambda: losses.SubCenterArcFaceLoss(
        num_classes=3, embedding_size=8
    ),
    "ThresholdConsistentMarginLoss": lambda: losses.ThresholdConsistentMarginLoss(),
}


def batch(dtype, size=9, generator_seed=3):
    generator = torch.Generator().manual_seed(generator_seed)
    embeddings = torch.randn(size, 8, generator=generator, dtype=dtype)
    labels = torch.arange(size) % 3
    return (
        embeddings.to(TEST_DEVICE).requires_grad_(),
        labels.to(TEST_DEVICE),
    )


def references(dtype):
    generator = torch.Generator().manual_seed(8)
    ref_emb = torch.randn(7, 8, generator=generator, dtype=dtype)
    return ref_emb.to(TEST_DEVICE).requires_grad_(), (torch.arange(7) % 3).to(
        TEST_DEVICE
    )


def forbid_distance(module):
    def forbidden(query_emb, ref_emb=None):
        raise AssertionError("mat must replace the distance call")

    module.distance.forward = forbidden


def value_and_gradients(loss, inputs):
    gradients = torch.autograd.grad(
        loss, inputs, allow_unused=True, materialize_grads=True
    )
    return loss.detach(), gradients


def assert_same(test, first, second):
    (first_loss, first_gradients), (second_loss, second_gradients) = first, second
    torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=0)
    for x, y in zip(first_gradients, second_gradients, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    test.assertTrue(first_loss.isfinite())


class TestLossMatrixInput(unittest.TestCase):
    def test_support_declarations_cover_every_exported_loss(self):
        exported = {
            name
            for name, value in vars(losses).items()
            if inspect.isclass(value)
            and issubclass(value, losses.BaseMetricLossFunction)
            and value is not losses.BaseMetricLossFunction
            and name not in ("BaseProxyLoss", "GenericPairLoss", "MultiSupConLoss")
        }
        supported = {*REFERENCE_LOSSES, *BATCH_ONLY_LOSSES}
        self.assertEqual(exported, supported | UNSUPPORTED_LOSSES.keys())
        for name in supported:
            self.assertTrue(getattr(losses, name).supports_mat, name)
        for name in UNSUPPORTED_LOSSES:
            self.assertFalse(getattr(losses, name).supports_mat, name)

    def test_batch_matrix_matches_recomputation(self):
        for name, factory in {**REFERENCE_LOSSES, **BATCH_ONLY_LOSSES}.items():
            for dtype in TEST_DTYPES:
                with self.subTest(loss=name, dtype=dtype):
                    first = factory().to(TEST_DEVICE, dtype)
                    second = copy.deepcopy(first)
                    embeddings, labels = batch(dtype)
                    expected = value_and_gradients(
                        first(embeddings, labels), embeddings
                    )
                    mat = second.distance(embeddings)
                    forbid_distance(second)
                    version = mat._version
                    before = mat.detach().clone()
                    actual = value_and_gradients(
                        second(embeddings, labels, mat=mat), embeddings
                    )
                    assert_same(self, actual, expected)
                    self.assertEqual(mat._version, version)
                    self.assertTrue(torch.equal(mat.detach(), before))

    def test_reference_matrix_matches_recomputation(self):
        for name, factory in REFERENCE_LOSSES.items():
            for dtype in TEST_DTYPES:
                with self.subTest(loss=name, dtype=dtype):
                    first = factory().to(TEST_DEVICE, dtype)
                    second = copy.deepcopy(first)
                    embeddings, labels = batch(dtype)
                    ref_emb, ref_labels = references(dtype)
                    inputs = (embeddings, ref_emb)
                    expected = value_and_gradients(
                        first(
                            embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels
                        ),
                        inputs,
                    )
                    mat = second.distance(embeddings, ref_emb)
                    forbid_distance(second)
                    actual = value_and_gradients(
                        second(
                            embeddings,
                            labels,
                            ref_emb=ref_emb,
                            ref_labels=ref_labels,
                            mat=mat,
                        ),
                        inputs,
                    )
                    assert_same(self, actual, expected)
                    self.assertTrue(expected[1][1].any())

    def test_multilabel_matrix_matches_recomputation(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                loss_func = losses.MultiSupConLoss()
                embeddings, _ = batch(dtype)
                labels = torch.eye(3, device=TEST_DEVICE)[torch.arange(9) % 3].bool()
                ref_emb, _ = references(dtype)
                ref_labels = torch.eye(3, device=TEST_DEVICE)[
                    torch.arange(7) % 3
                ].bool()
                expected = loss_func(
                    embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels
                )
                mat = loss_func.distance(embeddings, ref_emb)
                forbid_distance(loss_func)
                actual = loss_func(
                    embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels, mat=mat
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_label_free_soft_margin_reads_mat(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                first = losses.DynamicSoftMarginLoss().to(TEST_DEVICE, dtype)
                second = copy.deepcopy(first)
                embeddings, _ = batch(dtype)
                ref_emb = embeddings.detach().flip(0).requires_grad_()
                inputs = (embeddings, ref_emb)
                expected = value_and_gradients(
                    first(embeddings, ref_emb=ref_emb), inputs
                )
                mat = second.distance(embeddings, ref_emb)
                forbid_distance(second)
                actual = value_and_gradients(
                    second(embeddings, ref_emb=ref_emb, mat=mat), inputs
                )
                assert_same(self, actual, expected)

    def test_indices_tuple_selects_from_mat(self):
        loss_func = losses.ContrastiveLoss()
        embeddings, labels = batch(torch.float64)
        pairs = miners.MultiSimilarityMiner(epsilon=0.3)(embeddings, labels)
        expected = loss_func(embeddings, labels, pairs)
        mat = loss_func.distance(embeddings)
        actual = loss_func(embeddings, labels, pairs, mat=mat)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_loss_terms_read_mat(self):
        loss_func = losses.ContrastiveLoss()
        embeddings, labels = batch(torch.float64)
        expected = loss_func.loss_terms(embeddings, labels)
        actual = loss_func.loss_terms(
            embeddings, labels, mat=loss_func.distance(embeddings)
        )
        self.assertEqual(expected.keys(), actual.keys())
        for name, term in expected.items():
            torch.testing.assert_close(actual[name]["losses"], term["losses"])

    def test_per_batch_metadata_reaches_the_loss(self):
        embeddings, labels = batch(torch.float64)
        rewards = torch.linspace(
            0.5, 2.0, 9, dtype=torch.float64, device=TEST_DEVICE, requires_grad=True
        )

        class Scaled(distances.DotProductSimilarity):
            rewards: torch.Tensor

            def compute_mat(self, query_emb, ref_emb):
                return (query_emb @ ref_emb.T) * self.rewards

        mutated = Scaled()
        mutated.rewards = rewards
        expected = losses.NTXentLoss(distance=mutated)(embeddings, labels)

        distance = distances.DotProductSimilarity()
        mat = (embeddings @ embeddings.T) * rewards
        actual = losses.NTXentLoss(distance=distance)(embeddings, labels, mat=mat)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        reward_gradient = torch.autograd.grad(actual, rewards)[0]
        self.assertTrue(reward_gradient.any())

    def test_loss_needs_a_tracked_matrix(self):
        loss_func = losses.ContrastiveLoss()
        embeddings, labels = batch(torch.float64)
        with torch.no_grad():
            mat = loss_func.distance(embeddings)
        with self.assertRaisesRegex(ValueError, "no autograd graph"):
            loss_func(embeddings, labels, mat=mat)
        with self.assertRaisesRegex(ValueError, "no autograd graph"):
            loss_func(embeddings, labels, mat=mat.detach())

    def test_untracked_matrix_is_valid_when_nothing_trains(self):
        loss_func = losses.ContrastiveLoss()
        embeddings, labels = batch(torch.float64)
        embeddings = embeddings.detach()
        expected = loss_func(embeddings, labels)
        actual = loss_func(embeddings, labels, mat=loss_func.distance(embeddings))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with torch.no_grad():
            tracked = embeddings.clone().requires_grad_()
            result = loss_func(tracked, labels, mat=loss_func.distance(embeddings))
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_invalid_matrices(self):
        loss_func = losses.ContrastiveLoss()
        embeddings, labels = batch(torch.float64)
        ref_emb, ref_labels = references(torch.float64)
        mat = loss_func.distance(embeddings)
        cases = [
            (mat[:4], {}, "shape"),
            (mat[0], {}, "shape"),
            (mat, {"ref_emb": ref_emb, "ref_labels": ref_labels}, "shape"),
            (mat.long(), {}, "floating-point"),
        ]
        for bad, options, message in cases:
            with (
                self.subTest(message=message, shape=tuple(bad.shape)),
                self.assertRaisesRegex(ValueError, message),
            ):
                loss_func(embeddings, labels, mat=bad, **options)

    def test_wrong_device_is_rejected(self):
        loss_func = losses.ContrastiveLoss()
        embeddings, labels = batch(torch.float64)
        mat = loss_func.distance(embeddings).to("meta")
        with self.assertRaisesRegex(ValueError, "mat are on"):
            loss_func(embeddings, labels, mat=mat)

    def test_unsupported_losses_reject_mat(self):
        for name, factory in UNSUPPORTED_LOSSES.items():
            with self.subTest(loss=name):
                loss_func = factory().to(TEST_DEVICE)
                embeddings, labels = batch(torch.float32)
                mat = embeddings @ embeddings.T
                with self.assertRaisesRegex(ValueError, "cannot use mat"):
                    loss_func(embeddings, labels, mat=mat)

    def test_batch_only_losses_keep_rejecting_references(self):
        for name, factory in BATCH_ONLY_LOSSES.items():
            with self.subTest(loss=name):
                loss_func = factory().to(TEST_DEVICE)
                embeddings, labels = batch(torch.float32)
                ref_emb, ref_labels = references(torch.float32)
                with self.assertRaises(ValueError):
                    loss_func(
                        embeddings,
                        labels,
                        ref_emb=ref_emb,
                        ref_labels=ref_labels,
                        mat=embeddings @ ref_emb.T,
                    )

    def test_swap_is_rejected(self):
        loss_func = losses.TripletMarginLoss(swap=True)
        embeddings, labels = batch(torch.float64)
        with self.assertRaisesRegex(ValueError, "swap"):
            loss_func(embeddings, labels, mat=loss_func.distance(embeddings))

    def test_wrappers_have_no_matrix_input(self):
        wrapped = losses.ContrastiveLoss()
        modules = {
            "CrossBatchMemory": losses.CrossBatchMemory(wrapped, embedding_size=8),
            "MultipleLosses": losses.MultipleLosses({"loss": wrapped}),
            "SelfSupervisedLoss": losses.SelfSupervisedLoss(wrapped),
            "DistributedLossWrapper": DistributedLossWrapper(wrapped),
            "DistributedMinerWrapper": DistributedMinerWrapper(miners.HDCMiner()),
        }
        for name, module in modules.items():
            with self.subTest(module=name):
                self.assertNotIn("mat", inspect.signature(module.forward).parameters)

    def test_no_module_retains_the_matrix(self):
        embeddings, labels = batch(torch.float64)
        distance = distances.LpDistance()
        loss_func = losses.ContrastiveLoss(distance=distance)
        miner = miners.PairMarginMiner(distance=distance)
        mat = distance(embeddings)
        reference = weakref.ref(mat)
        loss = loss_func(
            embeddings, labels, miner(embeddings, labels, mat=mat), mat=mat
        )
        loss.backward()
        del mat, loss
        gc.collect()
        self.assertIsNone(reference())

    def test_shared_matrix_matches_separate_computation(self):
        for dtype in TEST_DTYPES:
            for distance in (distances.LpDistance(), COSINE()):
                with self.subTest(dtype=dtype, distance=type(distance).__name__):
                    miner = miners.MultiSimilarityMiner(distance=distance)
                    first = losses.MultiSimilarityLoss(distance=distance)
                    second = losses.ContrastiveLoss(distance=distance)
                    embeddings, labels = batch(dtype)
                    pairs = miner(embeddings, labels)
                    separate = first(embeddings, labels, pairs) + second(
                        embeddings, labels, pairs
                    )
                    mat = distance(embeddings)
                    reused_pairs = miner(embeddings, labels, mat=mat)
                    shared = first(embeddings, labels, reused_pairs, mat=mat) + second(
                        embeddings, labels, reused_pairs, mat=mat
                    )
                    for x, y in zip(pairs, reused_pairs, strict=True):
                        self.assertTrue(torch.equal(x, y))
                    shared_result = value_and_gradients(shared, embeddings)
                    separate_result = value_and_gradients(separate, embeddings)
                    helpers.assert_close(shared_result[0], separate_result[0], dtype)
                    helpers.assert_close(
                        shared_result[1][0], separate_result[1][0], dtype
                    )

    def test_separate_backward_calls_need_retained_graph(self):
        loss_func = losses.ContrastiveLoss()
        embeddings, labels = batch(torch.float64)
        mat = loss_func.distance(embeddings)
        loss_func(embeddings, labels, mat=mat).backward()
        second = loss_func(embeddings, labels, mat=mat)
        with self.assertRaisesRegex(RuntimeError, "second time"):
            second.backward()


class TestMinerMatrixInput(unittest.TestCase):
    def test_support_declarations_cover_every_exported_miner(self):
        exported = {
            name
            for name, value in vars(miners).items()
            if inspect.isclass(value)
            and issubclass(value, miners.BaseMiner)
            and value is not miners.BaseMiner
        }
        self.assertEqual(exported, {*MINERS, "AngularMiner"})
        self.assertFalse(miners.AngularMiner.supports_mat)

    def test_batch_matrix_matches_recomputation(self):
        for name, factory in MINERS.items():
            for dtype in TEST_DTYPES:
                with self.subTest(miner=name, dtype=dtype):
                    miner = factory().to(TEST_DEVICE, dtype)
                    embeddings, labels = batch(dtype)
                    torch.manual_seed(1)
                    expected = miner(embeddings, labels)
                    mat = miner.distance(embeddings)
                    forbid_distance(miner)
                    version = mat._version
                    torch.manual_seed(1)
                    actual = miner(embeddings, labels, mat=mat)
                    self.assertEqual(mat._version, version)
                    self.assertEqual(type(actual), type(expected))
                    for x, y in zip(actual, expected, strict=True):
                        self.assertTrue(torch.equal(x, y))

    def test_reference_matrix_matches_recomputation(self):
        for name, factory in MINERS.items():
            for dtype in TEST_DTYPES:
                with self.subTest(miner=name, dtype=dtype):
                    miner = factory().to(TEST_DEVICE, dtype)
                    embeddings, labels = batch(dtype)
                    ref_emb, ref_labels = references(dtype)
                    torch.manual_seed(1)
                    expected = miner(embeddings, labels, None, ref_emb, ref_labels)
                    mat = miner.distance(embeddings, ref_emb)
                    forbid_distance(miner)
                    torch.manual_seed(1)
                    actual = miner(
                        embeddings, labels, None, ref_emb, ref_labels, mat=mat
                    )
                    for x, y in zip(actual, expected, strict=True):
                        self.assertTrue(torch.equal(x, y))

    def test_matrix_needs_no_graph(self):
        miner = miners.PairMarginMiner()
        embeddings, labels = batch(torch.float64)
        with torch.no_grad():
            mat = miner.distance(embeddings)
        expected = miner(embeddings, labels)
        for x, y in zip(miner(embeddings, labels, mat=mat), expected, strict=True):
            self.assertTrue(torch.equal(x, y))

    def test_invalid_input(self):
        miner = miners.PairMarginMiner()
        embeddings, labels = batch(torch.float64)
        with self.assertRaisesRegex(ValueError, "shape"):
            miner(embeddings, labels, mat=torch.zeros(2, 2, device=TEST_DEVICE))
        with self.assertRaisesRegex(ValueError, "cannot use mat"):
            miners.AngularMiner()(embeddings, labels, mat=miner.distance(embeddings))
