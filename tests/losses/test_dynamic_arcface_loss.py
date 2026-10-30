import math
import unittest

import torch
import torch.nn.functional as F

from pytorch_metric_learning.losses import ArcFaceLoss, DynamicArcFaceLoss
from pytorch_metric_learning.types import SubCenterOutliers

from .. import TEST_DEVICE, TEST_DTYPES

COUNTS = [1, 16, 81, 10000]


def paper_margin(count, lower=0.05, upper=0.5, power=0.25, smallest=1, largest=10000):
    a = (upper - lower) / (smallest**-power - largest**-power)
    b = upper - a * smallest**-power
    return a * count**-power + b


def reference_loss(embeddings, weight, labels, scale, sub_centers):
    cosine = F.normalize(embeddings, dim=1) @ F.normalize(weight, dim=1).T
    cosine = cosine.view(len(labels), len(COUNTS), sub_centers).max(dim=2).values
    losses = []
    for row, label in zip(cosine, labels.tolist(), strict=True):
        margin = paper_margin(COUNTS[label])
        theta = torch.acos(row[label])
        if theta <= math.pi - margin:
            target = torch.cos(theta + margin)
        else:
            target = torch.cos(theta) - margin * math.sin(margin)
        logits = torch.cat([row[:label], target.unsqueeze(0), row[label + 1 :]])
        losses.append(-F.log_softmax(logits * scale, dim=0)[label])
    return torch.stack(losses).mean()


class TestDynamicArcFaceLoss(unittest.TestCase):
    def test_margins_follow_power_law_between_bounds(self):
        loss_func = DynamicArcFaceLoss(COUNTS, 8).to(TEST_DEVICE, torch.float64)
        expected = torch.tensor(
            [paper_margin(count) for count in COUNTS],
            dtype=torch.float64,
            device=TEST_DEVICE,
        )
        self.assertTrue(torch.allclose(loss_func.class_margins, expected, atol=1e-6))
        self.assertAlmostEqual(loss_func.class_margins[0].item(), 0.5, places=6)
        self.assertAlmostEqual(loss_func.class_margins[-1].item(), 0.05, places=6)

    def test_bounds_and_power_are_configurable(self):
        margins = DynamicArcFaceLoss(
            [1, 4, 16], 8, margin_bounds=(0.1, 0.4), power=0.5
        ).class_margins
        expected = torch.tensor([0.4, 0.2, 0.1])
        self.assertTrue(torch.allclose(margins, expected, atol=1e-6))

    def test_values_and_gradients_match_reference(self):
        torch.manual_seed(452)
        labels = torch.tensor([0, 1, 2, 3, 0, 3], device=TEST_DEVICE)
        for sub_centers in (1, 3):
            loss_func = DynamicArcFaceLoss(COUNTS, 8, sub_centers=sub_centers).to(
                TEST_DEVICE, torch.float64
            )
            embeddings = torch.randn(
                6, 8, dtype=torch.float64, device=TEST_DEVICE, requires_grad=True
            )
            loss = loss_func(embeddings, labels)
            expected = reference_loss(
                embeddings, loss_func.weight, labels, 64, sub_centers
            )
            self.assertAlmostEqual(loss.item(), expected.item(), places=5)
            gradient = torch.autograd.grad(loss, embeddings)[0]
            expected_gradient = torch.autograd.grad(expected, embeddings)[0]
            self.assertTrue(torch.allclose(gradient, expected_gradient, atol=1e-3))

    def test_large_angles_use_each_class_fallback(self):
        loss_func = DynamicArcFaceLoss(COUNTS, 2, sub_centers=1).to(
            TEST_DEVICE, torch.float64
        )
        angle = math.pi - 0.2
        cosine = torch.tensor(
            [math.cos(angle)] * 2, dtype=torch.float64, device=TEST_DEVICE
        )
        labels = torch.tensor([0, 3], device=TEST_DEVICE)
        modified = loss_func.modify_cosine_of_target_classes(cosine, labels)
        margin = paper_margin(1)
        self.assertAlmostEqual(
            modified[0].item(), math.cos(angle) - margin * math.sin(margin), places=6
        )
        self.assertAlmostEqual(
            modified[1].item(), math.cos(angle + paper_margin(10000)), places=6
        )

    def test_each_class_matches_arcface_with_its_margin(self):
        torch.manual_seed(5)
        for dtype in TEST_DTYPES:
            dynamic = DynamicArcFaceLoss(COUNTS, 8, sub_centers=1).to(
                TEST_DEVICE, dtype
            )
            embeddings = torch.randn(6, 8, device=TEST_DEVICE, dtype=dtype)
            for label, margin in enumerate(dynamic.class_margins.tolist()):
                constant = ArcFaceLoss(len(COUNTS), 8, margin=margin).to(
                    TEST_DEVICE, dtype
                )
                constant.load_state_dict(dynamic.state_dict())
                labels = torch.full((6,), label, device=TEST_DEVICE)
                self.assertAlmostEqual(
                    dynamic(embeddings, labels).item(),
                    constant(embeddings, labels).item(),
                    delta=0.2 if dtype == torch.float16 else 1e-4,
                )

    def test_rarer_classes_are_penalized_more(self):
        loss_func = DynamicArcFaceLoss([1, 10000], 2, scale=1, sub_centers=1).to(
            TEST_DEVICE
        )
        loss_func.weight.data = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], device=TEST_DEVICE
        )
        rare = torch.tensor([[1.0, 0.2]], device=TEST_DEVICE)
        common = torch.tensor([[0.2, 1.0]], device=TEST_DEVICE)
        self.assertGreater(
            loss_func(rare, torch.tensor([0], device=TEST_DEVICE)).item(),
            loss_func(common, torch.tensor([1], device=TEST_DEVICE)).item(),
        )

    def test_forward_does_not_change_module_state(self):
        loss_func = DynamicArcFaceLoss(COUNTS, 8).to(TEST_DEVICE)
        margins = loss_func.class_margins.clone()
        margin = loss_func.margin
        embeddings = torch.randn(4, 8, device=TEST_DEVICE)
        loss_func(embeddings, torch.arange(4, device=TEST_DEVICE))
        self.assertEqual(loss_func.margin, margin)
        self.assertTrue(torch.equal(loss_func.class_margins, margins))

    def test_state_dict_holds_only_learned_weights(self):
        loss_func = DynamicArcFaceLoss(COUNTS, 8)
        self.assertEqual(list(loss_func.state_dict()), ["weight"])
        restored = DynamicArcFaceLoss(COUNTS, 8)
        restored.load_state_dict(loss_func.state_dict())
        self.assertTrue(torch.equal(restored.class_margins, loss_func.class_margins))

    def test_margins_follow_module_placement(self):
        for dtype in TEST_DTYPES:
            loss_func = DynamicArcFaceLoss(COUNTS, 8).to(TEST_DEVICE, dtype)
            self.assertEqual(loss_func.class_margins.dtype, dtype)
            self.assertEqual(loss_func.class_margins.device.type, TEST_DEVICE.type)

    def test_unplaced_module_raises(self):
        loss_func = DynamicArcFaceLoss(COUNTS, 8).to(TEST_DEVICE, torch.float32)
        embeddings = torch.randn(4, 8, device=TEST_DEVICE, dtype=torch.float64)
        with self.assertRaises(ValueError):
            loss_func(embeddings, torch.arange(4, device=TEST_DEVICE))

    def test_tensor_counts_match_sequence_counts(self):
        margins = DynamicArcFaceLoss(torch.tensor(COUNTS), 8).class_margins
        self.assertTrue(
            torch.equal(margins, DynamicArcFaceLoss(COUNTS, 8).class_margins)
        )

    def test_invalid_arguments(self):
        for counts in ([], [0, 3], [-1, 3], [5, 5], [[1, 2]]):
            with self.assertRaises(ValueError):
                DynamicArcFaceLoss(torch.tensor(counts), 8)
        for counts in ([1.5, 2.5], [True, False]):
            with self.assertRaises(TypeError):
                DynamicArcFaceLoss(torch.tensor(counts), 8)
        for bounds in ((0.5, 0.5), (0.5, 0.1), (-0.1, 0.5)):
            with self.assertRaises(ValueError):
                DynamicArcFaceLoss(COUNTS, 8, margin_bounds=bounds)
        with self.assertRaises(ValueError):
            DynamicArcFaceLoss(COUNTS, 8, power=0)

    def test_subcenter_outliers_and_logits_are_inherited(self):
        loss_func = DynamicArcFaceLoss(COUNTS, 8).to(TEST_DEVICE)
        embeddings = torch.randn(4, 8, device=TEST_DEVICE)
        labels = torch.arange(4, device=TEST_DEVICE)
        result = loss_func.get_outliers(embeddings, labels)
        self.assertIsInstance(result, SubCenterOutliers)
        self.assertEqual(loss_func.get_logits(embeddings).shape, (4, 4))
