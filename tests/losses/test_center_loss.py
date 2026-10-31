import copy
import unittest

import torch

from pytorch_metric_learning import losses, reducers

from .. import TEST_DEVICE, TEST_DTYPES, helpers


def loss_with_centers(centers):
    loss_func = losses.CenterLoss(
        num_classes=centers.shape[0], embedding_size=centers.shape[1]
    ).to(TEST_DEVICE, centers.dtype)
    loss_func.centers.copy_(centers)
    return loss_func


def batch(dtype=torch.float64, size=12, classes=4, dimensions=3):
    generator = torch.Generator().manual_seed(6)
    x = torch.randn(size, dimensions, generator=generator, dtype=torch.float64)
    centers = torch.randn(classes, dimensions, generator=generator, dtype=torch.float64)
    labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2, 0, 2, 1, 0][:size])
    return (
        x.to(TEST_DEVICE, dtype),
        centers.to(TEST_DEVICE, dtype),
        labels.to(TEST_DEVICE),
    )


def equation_2(x, centers, labels):
    terms = [
        (x[i] - centers[int(labels[i])]).pow(2).sum() / 2 for i in range(len(labels))
    ]
    return torch.stack(terms).sum()


def equation_4(x, centers, labels, alpha):
    updated = centers.clone()
    for j in range(len(centers)):
        rows = [i for i in range(len(labels)) if labels[i] == j]
        if rows:
            total = sum(centers[j] - x[i] for i in rows)
            updated[j] = centers[j] - alpha * total / (1 + len(rows))
    return updated


class TestCenterLoss(unittest.TestCase):
    def test_equation_2_value_and_gradient(self):
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                x, centers, labels = batch(dtype)
                loss_func = loss_with_centers(centers)
                loss_func.reducer = reducers.SumReducer()
                helpers.assert_matches_float64(
                    lambda v, f=loss_func, y=labels: f(v, y),
                    lambda v, c=centers, y=labels: equation_2(v, c.double(), y),
                    (x,),
                    dtype,
                )

    def test_default_reducer_averages(self):
        x, centers, labels = batch()
        loss = loss_with_centers(centers)(x, labels)
        self.assertAlmostEqual(
            loss.item(), equation_2(x, centers, labels).item() / len(x)
        )

    def test_gradient_is_equation_3(self):
        x, centers, labels = batch()
        x = x.requires_grad_()
        loss_func = loss_with_centers(centers)
        loss_func.reducer = reducers.SumReducer()
        loss_func(x, labels).backward()
        assert x.grad is not None
        torch.testing.assert_close(x.grad, x.detach() - centers[labels])

    def test_update_is_equation_4(self):
        for dtype in (torch.float32, torch.float64):
            for alpha in (0.0, 0.1, 0.5, 1.0):
                with self.subTest(dtype=dtype, alpha=alpha):
                    x, centers, labels = batch(dtype)
                    loss_func = loss_with_centers(centers)
                    loss_func.alpha = alpha
                    loss_func.update_centers(x, labels)
                    torch.testing.assert_close(
                        loss_func.centers, equation_4(x, centers, labels, alpha)
                    )

    def test_one_sample_moves_its_center_by_half_of_alpha(self):
        centers = torch.tensor([[0.0, 0.0]], dtype=torch.float64, device=TEST_DEVICE)
        x = torch.tensor([[4.0, 8.0]], dtype=torch.float64, device=TEST_DEVICE)
        loss_func = loss_with_centers(centers)
        loss_func.alpha = 1.0
        loss_func.update_centers(x, torch.tensor([0], device=TEST_DEVICE))
        self.assertEqual(loss_func.centers.tolist(), [[2.0, 4.0]])

    def test_update_is_not_a_gradient_step(self):
        x, centers, labels = batch()
        loss_func = loss_with_centers(centers)
        loss_func.update_centers(x, labels)
        counts = torch.bincount(labels, minlength=4).unsqueeze(1).double()
        offsets = torch.zeros_like(centers).index_add(0, labels, centers[labels] - x)
        gradient_step = centers - 0.5 * offsets / counts.clamp_min(1)
        self.assertFalse(torch.allclose(loss_func.centers, gradient_step))

    def test_classes_outside_the_batch_stay_put(self):
        x, centers, labels = batch()
        loss_func = loss_with_centers(centers)
        loss_func.update_centers(x, labels)
        self.assertTrue(torch.equal(loss_func.centers[3], centers[3]))
        self.assertFalse(torch.equal(loss_func.centers[0], centers[0]))

    def test_repeated_updates_reach_the_class_mean(self):
        x, centers, labels = batch()
        loss_func = loss_with_centers(centers)
        for _ in range(300):
            loss_func.update_centers(x, labels)
        means = torch.stack([x[labels == j].mean(dim=0) for j in range(3)])
        torch.testing.assert_close(loss_func.centers[:3], means)

    def test_forward_does_not_move_centers(self):
        x, centers, labels = batch()
        loss_func = loss_with_centers(centers)
        loss_func(x, labels)
        loss_func.loss_terms(x, labels)
        self.assertTrue(torch.equal(loss_func.centers, centers))

    def test_no_gradient_reaches_the_centers(self):
        x, centers, labels = batch()
        x = x.requires_grad_()
        loss_func = loss_with_centers(centers)
        self.assertEqual(list(loss_func.parameters()), [])
        self.assertFalse(loss_func.centers.requires_grad)
        loss_func(x, labels).backward()
        self.assertIsNone(loss_func.centers.grad)

    def test_update_before_backward_is_safe_and_changes_nothing(self):
        x, centers, labels = batch()
        expected = x.clone().requires_grad_()
        loss_with_centers(centers)(expected, labels).backward()
        tracked = x.clone().requires_grad_()
        loss_func = loss_with_centers(centers)
        loss = loss_func(tracked, labels)
        loss_func.update_centers(tracked, labels)
        loss.backward()
        assert tracked.grad is not None and expected.grad is not None
        torch.testing.assert_close(tracked.grad, expected.grad)

    def test_update_keeps_the_embedding_graph(self):
        x, centers, labels = batch()
        x = x.requires_grad_()
        tracked = x * 2
        loss_func = loss_with_centers(centers)
        loss_func.update_centers(tracked, labels)
        self.assertFalse(loss_func.centers.requires_grad)
        self.assertIsNotNone(tracked.grad_fn)

    def test_state_dict_round_trip(self):
        x, centers, labels = batch()
        loss_func = loss_with_centers(centers)
        loss_func.update_centers(x, labels)
        restored = copy.deepcopy(loss_with_centers(centers))
        restored.load_state_dict(loss_func.state_dict())
        self.assertEqual(set(loss_func.state_dict()), {"centers"})
        self.assertTrue(torch.equal(restored.centers, loss_func.centers))

    def test_joint_training_shrinks_within_class_spread(self):
        torch.manual_seed(1)
        means = torch.tensor([[3.0, 0], [-3.0, 0], [0, 3.0], [0, -3.0]])
        labels = torch.arange(64) % 4
        data = means[labels] + torch.randn(64, 2)

        def train(weight):
            torch.manual_seed(2)
            model = torch.nn.Linear(2, 2)
            classifier = torch.nn.Linear(2, 4)
            center_loss = losses.CenterLoss(num_classes=4, embedding_size=2)
            parameters = [*model.parameters(), *classifier.parameters()]
            optimizer = torch.optim.SGD(parameters, lr=0.1)
            for _ in range(300):
                embeddings = model(data)
                loss = torch.nn.functional.cross_entropy(classifier(embeddings), labels)
                loss = loss + weight * center_loss(embeddings, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                center_loss.update_centers(embeddings, labels)
            embeddings = model(data).detach()
            within = torch.stack(
                [embeddings[labels == j].std(dim=0).sum() for j in range(4)]
            ).mean()
            spread = embeddings.std(dim=0).sum()
            return (within / spread).item()

        self.assertLess(train(0.5), train(0.0))

    def test_low_precision_large_features_stay_finite(self):
        x = torch.full((4, 64), 200.0, dtype=torch.float16, device=TEST_DEVICE)
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        centers = torch.zeros(2, 64, dtype=torch.float16, device=TEST_DEVICE)
        loss_func = loss_with_centers(centers)
        loss = loss_func(x, labels)
        self.assertTrue(loss.isfinite())
        self.assertEqual(loss.item(), 200.0**2 * 64 / 2)
        loss_func.update_centers(x, labels)
        self.assertTrue(loss_func.centers.isfinite().all())
        self.assertEqual(loss_func.centers.dtype, torch.float16)

    def test_empty_batch(self):
        x, centers, _ = batch()
        loss_func = loss_with_centers(centers)
        empty = x[:0]
        labels = torch.empty(0, dtype=torch.long, device=TEST_DEVICE)
        self.assertEqual(loss_func(empty, labels).item(), 0)
        loss_func.update_centers(empty, labels)
        self.assertTrue(torch.equal(loss_func.centers, centers))

    def test_invalid_arguments(self):
        for alpha in (-0.1, 1.1):
            with (
                self.subTest(alpha=alpha),
                self.assertRaisesRegex(ValueError, "alpha"),
            ):
                losses.CenterLoss(4, 3, alpha=alpha)
        x, centers, labels = batch()
        loss_func = loss_with_centers(centers)
        for method in (loss_func, loss_func.update_centers):
            with self.assertRaisesRegex(ValueError, "num_classes"):
                method(x, labels + 4)
            with self.assertRaisesRegex(ValueError, "num_classes"):
                method(x, labels - 1)
            with self.assertRaisesRegex(ValueError, "embedding_size"):
                method(x[:, :2], labels)
            with self.assertRaises(ValueError):
                method(x, labels[:3])
        with self.assertRaisesRegex(ValueError, "Move the loss"):
            losses.CenterLoss(4, 3).to(TEST_DEVICE)(x, labels)

    def test_rejects_unsupported_inputs(self):
        x, centers, labels = batch()
        loss_func = loss_with_centers(centers)
        pairs = tuple(torch.tensor([0], device=TEST_DEVICE) for _ in range(4))
        with self.assertRaises(ValueError):
            loss_func(x, labels, pairs)
        with self.assertRaises(ValueError):
            loss_func(x, labels, ref_emb=x, ref_labels=labels)
        with self.assertRaises(ValueError):
            loss_func(x)
