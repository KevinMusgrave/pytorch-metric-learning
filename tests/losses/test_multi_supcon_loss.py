import math
import unittest

import torch

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.distributed import DistributedLossWrapper
from pytorch_metric_learning.losses import (
    CrossBatchMemory,
    MultipleLosses,
    MultiSupConLoss,
    SelfSupervisedLoss,
    SupConLoss,
)
from pytorch_metric_learning.miners import BatchHardMiner
from pytorch_metric_learning.reducers import MeanReducer, SumReducer

from .. import TEST_DEVICE, TEST_DTYPES

EMBEDDINGS = [
    [0.1, 0.3, 0.1],
    [0.23, -0.2, -0.1],
    [0.1, -0.16, 0.1],
    [0.13, -0.13, 0.2],
]
LABELS = [[1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]]


def jaccard(first, second):
    first = {i for i, value in enumerate(first) if value}
    second = {i for i, value in enumerate(second) if value}
    if not first and not second:
        return 1.0
    return len(first & second) / len(first | second)


def reference_anchor_losses(
    embeddings, labels, ref_emb, ref_labels, temperature, threshold, self_excluded
):
    normalized = torch.nn.functional.normalize(embeddings, dim=1)
    normalized_ref = torch.nn.functional.normalize(ref_emb, dim=1)
    losses = {}
    for i in range(len(embeddings)):
        others = [a for a in range(len(ref_emb)) if not (self_excluded and a == i)]
        logits = [(normalized[i] @ normalized_ref[a]) / temperature for a in others]
        total = torch.logsumexp(torch.stack(logits), dim=0) if logits else None
        anchor = []
        for a, logit in zip(others, logits, strict=True):
            similarity = jaccard(labels[i], ref_labels[a])
            if similarity >= threshold:
                anchor.append(-similarity * (logit - total))
        if anchor:
            losses[i] = sum(anchor) / len(anchor)
    return losses


def reference_loss(
    embeddings,
    labels,
    temperature=0.1,
    threshold=0.5,
    ref_emb=None,
    ref_labels=None,
):
    separate = ref_emb is not None
    losses = reference_anchor_losses(
        embeddings,
        labels,
        ref_emb if separate else embeddings,
        ref_labels if separate else labels,
        temperature,
        threshold,
        not separate,
    )
    nonzero = [value for value in losses.values() if value != 0]
    return sum(nonzero) / len(nonzero)


def tolerance(dtype):
    return 2e-2 if dtype == torch.float16 else 1e-5


class TestMultiSupConLoss(unittest.TestCase):
    def inputs(self, dtype):
        embeddings = torch.tensor(
            EMBEDDINGS, dtype=dtype, device=TEST_DEVICE, requires_grad=True
        )
        labels = torch.tensor(LABELS, dtype=torch.bool, device=TEST_DEVICE)
        return embeddings, labels

    def test_values_match_equation_three(self):
        for dtype in TEST_DTYPES:
            embeddings, labels = self.inputs(dtype)
            for temperature, threshold in ((0.07, 0.3), (0.1, 0.5), (0.5, 0.0)):
                loss = MultiSupConLoss(temperature, threshold)(embeddings, labels)
                expected = reference_loss(
                    embeddings.detach().to(torch.float64),
                    LABELS,
                    temperature,
                    threshold,
                )
                self.assertAlmostEqual(
                    loss.item(), expected.item(), delta=tolerance(dtype)
                )

    def test_value_matches_multisupcontrast(self):
        embeddings, labels = self.inputs(torch.float64)
        loss = MultiSupConLoss(temperature=0.07, threshold=0.3)(embeddings, labels)
        self.assertAlmostEqual(loss.item(), 3.2874, places=4)

    def test_gradient_matches_reference(self):
        embeddings, labels = self.inputs(torch.float64)
        MultiSupConLoss(0.07, 0.3)(embeddings, labels).backward()
        other = embeddings.detach().clone().requires_grad_()
        reference_loss(other, LABELS, 0.07, 0.3).backward()
        self.assertTrue(torch.allclose(embeddings.grad, other.grad, atol=1e-12))

    def test_float16_gradients_are_finite_and_follow_float64(self):
        embeddings, labels = self.inputs(torch.float16)
        MultiSupConLoss(0.07, 0.3)(embeddings, labels).backward()
        wide = embeddings.detach().to(torch.float64).requires_grad_()
        MultiSupConLoss(0.07, 0.3)(wide, labels).backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertTrue(
            torch.allclose(embeddings.grad.to(torch.float64), wide.grad, atol=2e-1)
        )

    def test_threshold_includes_exact_similarity(self):
        labels = torch.tensor(
            [[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.bool, device=TEST_DEVICE
        )
        a1, p, a2, n = MultiSupConLoss(threshold=0.5).pair_indices(labels)
        self.assertEqual(
            sorted(zip(a1.tolist(), p.tolist(), strict=True)), [(0, 1), (1, 0)]
        )
        self.assertEqual(
            sorted(zip(a2.tolist(), n.tolist(), strict=True)),
            [(0, 2), (1, 2), (2, 0), (2, 1)],
        )
        self.assertEqual(
            len(MultiSupConLoss(threshold=0.51).pair_indices(labels).pos_anchors), 0
        )

    def test_threshold_one_is_supcon(self):
        sets = torch.tensor([0, 0, 1, 1, 2, 3], device=TEST_DEVICE)
        multi_hot = torch.nn.functional.one_hot(sets).bool()
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(6, 4, dtype=dtype, device=TEST_DEVICE)
            loss = MultiSupConLoss(threshold=1)(embeddings, multi_hot)
            expected = SupConLoss()(embeddings, sets)
            self.assertAlmostEqual(loss.item(), expected.item(), delta=tolerance(dtype))

    def test_threshold_zero_counts_zero_weight_pairs(self):
        embeddings = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            dtype=torch.float64,
            device=TEST_DEVICE,
        )
        labels = torch.tensor(
            [[1, 0], [1, 0], [0, 1]], dtype=torch.bool, device=TEST_DEVICE
        )
        loss = MultiSupConLoss(temperature=1, threshold=0, reducer=SumReducer())
        log_01 = 0 - math.log(math.exp(0) + math.exp(-1))
        log_10 = 0 - math.log(math.exp(0) + math.exp(0))
        self.assertAlmostEqual(
            loss(embeddings, labels).item(), -(log_01 + log_10) / 2, places=10
        )

    def test_empty_label_sets_match_only_each_other(self):
        labels = torch.tensor(
            [[0, 0], [0, 0], [1, 0], [1, 0], [0, 1]],
            dtype=torch.bool,
            device=TEST_DEVICE,
        )
        pairs = MultiSupConLoss(threshold=0.5).pair_indices(labels)
        self.assertEqual(
            sorted(
                zip(pairs.pos_anchors.tolist(), pairs.positives.tolist(), strict=True)
            ),
            [(0, 1), (1, 0), (2, 3), (3, 2)],
        )
        self.assertEqual(int((pairs.neg_anchors == pairs.negatives).sum()), 0)
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(
                5, 3, dtype=dtype, device=TEST_DEVICE, requires_grad=True
            )
            loss = MultiSupConLoss(threshold=0.5)(embeddings, labels)
            (gradient,) = torch.autograd.grad(loss, embeddings)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(gradient).all())

    def test_anchors_without_positives_have_no_term(self):
        embeddings, labels = self.inputs(torch.float64)
        loss = MultiSupConLoss(threshold=0.5)
        anchors = loss.loss_terms(embeddings, labels)["loss"]["indices"]
        assert isinstance(anchors, torch.Tensor)
        self.assertEqual(anchors.tolist(), [0, 1, 2, 3])
        labels[1] = torch.tensor([0, 0, 0], device=TEST_DEVICE)
        anchors = loss.loss_terms(embeddings, labels)["loss"]["indices"]
        assert isinstance(anchors, torch.Tensor)
        self.assertEqual(anchors.tolist(), [2, 3])

    def test_batch_without_positives_is_differentiable_zero(self):
        for dtype in TEST_DTYPES:
            embeddings = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]],
                dtype=dtype,
                device=TEST_DEVICE,
                requires_grad=True,
            )
            labels = torch.eye(2, dtype=torch.bool, device=TEST_DEVICE)
            loss = MultiSupConLoss()(embeddings, labels)
            (gradient,) = torch.autograd.grad(loss, embeddings)
            self.assertEqual(loss.item(), 0)
            self.assertEqual(gradient.abs().sum().item(), 0)

    def test_empty_batch_is_differentiable_zero(self):
        embeddings = torch.empty(0, 3, device=TEST_DEVICE, requires_grad=True)
        labels = torch.empty(0, 2, dtype=torch.bool, device=TEST_DEVICE)
        loss = MultiSupConLoss()(embeddings, labels)
        loss.backward()
        self.assertEqual(loss.item(), 0)

    def test_references_keep_every_row(self):
        embeddings, labels = self.inputs(torch.float64)
        ref_emb = embeddings.detach().flip(0)
        ref_labels = labels.flip(0)
        for threshold in (0.0, 0.3):
            loss = MultiSupConLoss(0.1, threshold, reducer=MeanReducer())(
                embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels
            )
            anchors = reference_anchor_losses(
                embeddings,
                LABELS,
                ref_emb,
                ref_labels.tolist(),
                0.1,
                threshold,
                False,
            )
            expected = sum(anchors.values()) / len(anchors)
            self.assertAlmostEqual(loss.item(), expected.item(), places=10)

    def test_supplied_pairs_restrict_the_denominator(self):
        embeddings, labels = self.inputs(torch.float64)
        full = MultiSupConLoss(temperature=1, threshold=0.3)
        pairs = full.pair_indices(labels)
        self.assertEqual(
            full(embeddings, labels, pairs).item(), full(embeddings, labels).item()
        )
        anchors = torch.tensor([0], device=TEST_DEVICE)
        positives = torch.tensor([1], device=TEST_DEVICE)
        negatives = torch.tensor([3], device=TEST_DEVICE)
        loss = full(embeddings, labels, (anchors, positives, negatives))
        unit = torch.nn.functional.normalize(embeddings, dim=1)
        logits = torch.stack([unit[0] @ unit[1], unit[0] @ unit[3]])
        expected = -0.5 * (logits[0] - torch.logsumexp(logits, dim=0))
        self.assertAlmostEqual(loss.item(), expected.item(), places=10)

    def test_empty_supplied_pairs_select_nothing(self):
        embeddings, labels = self.inputs(torch.float32)
        empty = torch.empty(0, dtype=torch.long, device=TEST_DEVICE)
        loss = MultiSupConLoss()(embeddings, labels, (empty, empty, empty))
        self.assertEqual(loss.item(), 0)

    def test_inverted_distance_uses_negated_distances(self):
        embeddings, labels = self.inputs(torch.float64)
        loss = MultiSupConLoss(temperature=1, threshold=0.3, distance=LpDistance())
        unit = torch.nn.functional.normalize(embeddings, dim=1)
        distances = torch.cdist(unit, unit)
        logits = -distances[0, [1, 2, 3]]
        log_prob = logits - torch.logsumexp(logits, dim=0)
        expected = -(0.5 * log_prob[0] + log_prob[1] / 3) / 2
        terms = loss.loss_terms(embeddings, labels)["loss"]
        self.assertAlmostEqual(terms["losses"][0].item(), expected.item(), places=10)

    def test_invalid_inputs(self):
        embeddings, labels = self.inputs(torch.float32)
        loss = MultiSupConLoss()
        for threshold in (-0.1, 1.1, math.nan):
            with self.assertRaises(ValueError):
                MultiSupConLoss(threshold=threshold)
        with self.assertRaises(TypeError):
            loss(embeddings, labels.long())
        with self.assertRaises(TypeError):
            loss(embeddings, labels, ref_emb=embeddings, ref_labels=labels.float())
        with self.assertRaises(ValueError):
            loss(embeddings, labels[:, 0])
        with self.assertRaises(ValueError):
            loss(embeddings, labels, ref_emb=embeddings, ref_labels=labels[:, :2])
        with self.assertRaises(ValueError):
            loss(embeddings, labels[:2])
        with self.assertRaises(ValueError):
            loss(embeddings)
        with self.assertRaises(ValueError):
            loss(embeddings, labels.to("meta"))

    def test_loss_dtype_follows_embeddings(self):
        for dtype in TEST_DTYPES:
            embeddings, labels = self.inputs(dtype)
            self.assertEqual(MultiSupConLoss()(embeddings, labels).dtype, dtype)

    def test_unsupported_wrappers_reject_label_sets(self):
        embeddings, labels = self.inputs(torch.float32)
        with self.assertRaises(ValueError):
            SelfSupervisedLoss(MultiSupConLoss())(embeddings, embeddings)
        with self.assertRaises(ValueError):
            DistributedLossWrapper(MultiSupConLoss())(embeddings, labels)

    def test_multiple_losses_compose_label_set_losses(self):
        embeddings, labels = self.inputs(torch.float64)
        first, second = MultiSupConLoss(0.1, 0.3), MultiSupConLoss(0.2, 0.5)
        combined = MultipleLosses({"first": first, "second": second})(
            embeddings, labels
        )
        expected = first(embeddings, labels) + second(embeddings, labels)
        self.assertEqual(combined.item(), expected.item())


class TestMultiLabelCrossBatchMemory(unittest.TestCase):
    def memory(self, dtype=torch.float64, memory_size=16, miner=None, threshold=0.3):
        loss = MultiSupConLoss(temperature=0.07, threshold=threshold)
        return CrossBatchMemory(
            loss,
            embedding_size=3,
            memory_size=memory_size,
            miner=miner,
            label_dtype=torch.bool,
            label_shape=(3,),
            pair_function=loss.pair_indices,
        ).to(TEST_DEVICE, dtype)

    def inputs(self, dtype=torch.float64):
        embeddings = torch.tensor(
            EMBEDDINGS, dtype=dtype, device=TEST_DEVICE, requires_grad=True
        )
        labels = torch.tensor(LABELS, dtype=torch.bool, device=TEST_DEVICE)
        return embeddings, labels

    def test_queue_matches_stateless_reference(self):
        for dtype in TEST_DTYPES:
            memory = self.memory(dtype)
            embeddings, labels = self.inputs(dtype)
            for step in range(1, 4):
                loss = memory(embeddings, labels)
                ref_emb = embeddings.detach().repeat(step, 1)
                ref_labels = labels.repeat(step, 1)
                pairs = memory.loss.pair_indices(labels, ref_labels)
                keep = pairs.positives != pairs.pos_anchors
                expected = memory.loss(
                    embeddings,
                    labels,
                    (
                        pairs.pos_anchors[keep],
                        pairs.positives[keep],
                        pairs.neg_anchors,
                        pairs.negatives,
                    ),
                    ref_emb,
                    ref_labels,
                )
                self.assertAlmostEqual(loss.item(), expected.item(), places=2)
                self.assertEqual(memory.queue_idx, 4 * step)

    def test_queue_matches_multisupcontrast(self):
        embeddings, labels = self.inputs()
        expected = [3.2874, 4.3779, 4.4577, 4.5730, 4.6820, 4.7798, 4.8671]
        memory = self.memory(memory_size=1024)
        for value in expected:
            self.assertAlmostEqual(memory(embeddings, labels).item(), value, places=4)

    def test_empty_label_sets_exclude_their_own_slot(self):
        memory = self.memory(memory_size=4)
        embeddings = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float64,
            device=TEST_DEVICE,
            requires_grad=True,
        )
        labels = torch.zeros(2, 3, dtype=torch.bool, device=TEST_DEVICE)
        indices = memory.create_indices_tuple(
            embeddings,
            labels,
            embeddings.detach(),
            labels,
            None,
            torch.arange(2, device=TEST_DEVICE),
        )
        self.assertEqual(indices.pos_anchors.tolist(), [0, 1])
        self.assertEqual(indices.positives.tolist(), [1, 0])
        self.assertEqual(len(indices.neg_anchors), 0)

    def test_enqueue_mask_uses_rows_as_references(self):
        memory = self.memory()
        embeddings, labels = self.inputs()
        mask = torch.tensor([True, True, False, False], device=TEST_DEVICE)
        loss = memory(embeddings, labels, enqueue_mask=mask)
        expected = memory.loss(
            embeddings[~mask],
            labels[~mask],
            memory.loss.pair_indices(labels[~mask], labels[mask]),
            embeddings[mask].detach(),
            labels[mask],
        )
        self.assertAlmostEqual(loss.item(), expected.item(), places=10)
        self.assertEqual(memory.queue_idx, 2)

    def test_gradients_flow_only_through_anchors(self):
        memory = self.memory()
        embeddings, labels = self.inputs()
        memory(embeddings, labels).backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        self.assertGreater(embeddings.grad.abs().sum().item(), 0)
        self.assertFalse(memory.embedding_memory.requires_grad)

    def test_state_dict_restores_label_sets(self):
        memory = self.memory()
        embeddings, labels = self.inputs()
        memory(embeddings, labels)
        restored = self.memory()
        restored.load_state_dict(memory.state_dict())
        self.assertEqual(restored.queue_idx, 4)
        self.assertEqual(restored.label_memory.dtype, torch.bool)
        self.assertTrue(torch.equal(restored.label_memory, memory.label_memory))
        self.assertEqual(
            restored(embeddings, labels).item(), memory(embeddings, labels).item()
        )

    def test_rejected_calls_leave_queue_unchanged(self):
        embeddings, labels = self.inputs()
        memory = self.memory(miner=BatchHardMiner())
        with self.assertRaises(ValueError):
            memory(embeddings, labels)
        self.assertEqual(memory.queue_idx, 0)
        memory = self.memory()
        with self.assertRaises(ValueError):
            memory(embeddings, labels[:, :2])
        with self.assertRaises(ValueError):
            memory(embeddings, labels[:, 0])
        self.assertEqual(memory.queue_idx, 0)
        self.assertEqual(memory.label_memory.sum().item(), 0)

    def test_label_shape_requires_pair_function(self):
        with self.assertRaises(ValueError):
            CrossBatchMemory(MultiSupConLoss(), 3, label_shape=(3,))

    def test_long_memory_is_reported_by_the_loss(self):
        loss = MultiSupConLoss()
        memory = CrossBatchMemory(
            loss, 3, label_shape=(3,), pair_function=loss.pair_indices
        ).to(TEST_DEVICE, torch.float64)
        embeddings, labels = self.inputs()
        with self.assertRaises(TypeError):
            memory(embeddings, labels)
        self.assertEqual(memory.queue_idx, 0)
