import math
import unittest

import torch
from torch import nn

from docs.recipes.adversarial_metric_learning import (
    generator_losses,
    generator_step,
    metric_step,
)
from docs.recipes.cascaded_mining import cascaded_loss
from docs.recipes.memory_warmup import train_with_memory
from docs.recipes.mining_weights import subset_weights
from docs.recipes.npairs_class_mining import (
    candidate_examples,
    class_violations,
    draw_pairs,
    mined_batch,
    select_classes,
)
from docs.recipes.proxy_nca_plus_plus import (
    PoolAndEmbed,
    global_k_max_pool,
    make_optimizer,
    train,
)
from docs.recipes.training import train_metric, train_proxy, train_two_stream
from pytorch_metric_learning import distances, losses, miners
from pytorch_metric_learning.utils import indexing

from .. import TEST_DEVICE, TEST_DTYPES


class TestTrainingRecipes(unittest.TestCase):
    def test_optimization_updates_all_owners(self):
        for dtype in TEST_DTYPES:
            torch.manual_seed(17)
            data = torch.randn(12, 8, device=TEST_DEVICE, dtype=dtype)
            labels = torch.arange(4, device=TEST_DEVICE).repeat_interleave(3)
            for mode in ("metric", "classifier", "proxy", "two_stream"):
                with self.subTest(dtype=dtype, mode=mode):
                    model = nn.Linear(8, 4).to(device=TEST_DEVICE, dtype=dtype)
                    if mode == "proxy":
                        other = losses.NormalizedSoftmaxLoss(4, 4, temperature=0.1)
                    elif mode == "two_stream":
                        other = nn.Linear(8, 4)
                    else:
                        other = nn.Linear(4, 4)
                    other = other.to(device=TEST_DEVICE, dtype=dtype)
                    owners = [model] if mode == "metric" else [model, other]
                    before = [
                        next(owner.parameters()).detach().clone() for owner in owners
                    ]
                    if mode == "proxy":
                        assert isinstance(other, losses.NormalizedSoftmaxLoss)
                        history = train_proxy(model, other, data, labels)
                    elif mode == "two_stream":
                        history = train_two_stream(
                            model, other, data, labels, data[:8], labels[:8]
                        )
                    else:
                        history = train_metric(
                            model, data, labels, other if mode == "classifier" else None
                        )
                    self.assertLess(history[-1], history[0])
                    self.assertTrue(all(math.isfinite(value) for value in history))
                    for owner, initial in zip(owners, before, strict=True):
                        self.assertFalse(torch.equal(next(owner.parameters()), initial))

    def test_cascaded_candidates_and_reference_loss(self):
        torch.manual_seed(19)
        embeddings = torch.randn(8, 6, dtype=torch.float64, requires_grad=True)
        labels = torch.arange(4).repeat_interleave(2)
        stage_miners = tuple(miners.HDCMiner(p) for p in (1.0, 0.5, 0.5))
        loss, selected = cascaded_loss(
            embeddings,
            labels,
            [2, 2, 2],
            stage_miners,
            losses.ContrastiveLoss(neg_margin=1.4),
        )
        previous_pos = {
            (i, j)
            for i in range(8)
            for j in range(8)
            if i != j and labels[i] == labels[j]
        }
        previous_neg = {
            (i, j) for i in range(8) for j in range(8) if labels[i] != labels[j]
        }
        reference = embeddings.sum() * 0
        for stage, pairs, fraction in zip(
            embeddings.split(2, dim=1), selected, (1, 0.5, 0.5), strict=True
        ):
            a, p, b, n = pairs
            positive = set(zip(a.tolist(), p.tolist(), strict=True))
            negative = set(zip(b.tolist(), n.tolist(), strict=True))
            self.assertLessEqual(positive, previous_pos)
            self.assertLessEqual(negative, previous_neg)
            self.assertEqual(len(positive), math.ceil(len(previous_pos) * fraction))
            self.assertEqual(len(negative), math.ceil(len(previous_neg) * fraction))
            normalized = stage / stage.square().sum(dim=1, keepdim=True).sqrt()
            pos_dist = (normalized[a] - normalized[p]).square().sum(dim=1).sqrt()
            neg_dist = (normalized[b] - normalized[n]).square().sum(dim=1).sqrt()
            neg_loss = (1.4 - neg_dist).clamp_min(0)
            reference = (
                reference
                + pos_dist.mean()
                + neg_loss.sum() / (neg_loss > 0).sum().clamp_min(1)
            )
            previous_pos, previous_neg = positive, negative
        torch.testing.assert_close(loss, reference)
        actual_grad = torch.autograd.grad(loss, embeddings, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(reference, embeddings)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        fresh = embeddings.detach().clone().requires_grad_()
        repeated, _ = cascaded_loss(
            fresh,
            labels,
            [2, 2, 2],
            stage_miners,
            losses.ContrastiveLoss(neg_margin=1.4),
        )
        torch.testing.assert_close(repeated, loss)

    def test_adversarial_objectives_and_gradients(self):
        a = torch.tensor([[1.0, 0], [1, 0]], dtype=torch.float64)
        p = torch.tensor([[0.0, 1], [-1, 0]], dtype=torch.float64)
        n = torch.tensor([[1.0, 1], [0, -1]], dtype=torch.float64)
        synthetic = torch.tensor(
            [[-1.0, 0], [0, 1]], dtype=torch.float64, requires_grad=True
        )
        adv, hard, reg = generator_losses(nn.Identity(), a, p, n, synthetic)
        self.assertAlmostEqual(adv.item(), 2 - math.sqrt(2) - 0.2)
        self.assertAlmostEqual(hard.item(), 1.5)
        self.assertAlmostEqual(reg.item(), (6 + math.sqrt(2)) / 4)
        torch.testing.assert_close(
            torch.autograd.grad(adv, synthetic, retain_graph=True)[0],
            torch.zeros_like(synthetic),
        )
        torch.testing.assert_close(
            torch.autograd.grad(hard, synthetic, retain_graph=True)[0],
            torch.tensor([[0.0, 0], [-0.5, 0]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            torch.autograd.grad(reg, synthetic)[0],
            torch.tensor([[0.0, -math.sqrt(2) / 4], [0, 0]], dtype=torch.float64),
        )
        synthetic = torch.tensor(
            [[-0.6, 0.8], [0.2, 0.9]], dtype=torch.float64, requires_grad=True
        )
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x: generator_losses(nn.Identity(), a, p, n, x), synthetic
            )
        )

    def test_alternating_gradient_ownership(self):
        torch.manual_seed(23)
        trunk, embedder, generator = nn.Linear(4, 3), nn.Linear(3, 2), nn.Linear(9, 3)
        model_params = [*trunk.parameters(), *embedder.parameters()]
        model_optimizer = torch.optim.SGD(model_params, lr=0.1)
        generator_optimizer = torch.optim.SGD(generator.parameters(), lr=0.1)
        data = torch.randn(6, 4)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        triplets = indexing.get_all_triplets_indices(labels)
        for _ in range(2):
            model_optimizer.zero_grad(set_to_none=True)
            model_before = [x.detach().clone() for x in model_params]
            generator_before = generator.weight.detach().clone()
            loss = generator_step(
                trunk, embedder, generator, data, triplets, generator_optimizer
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertFalse(torch.equal(generator.weight, generator_before))
            for parameter, initial in zip(model_params, model_before, strict=True):
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, initial, rtol=0, atol=0)
            generator_optimizer.zero_grad(set_to_none=True)
            generator_before = generator.weight.detach().clone()
            loss = metric_step(
                trunk, embedder, generator, data, labels, triplets, model_optimizer
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertIsNone(generator.weight.grad)
            torch.testing.assert_close(
                generator.weight, generator_before, rtol=0, atol=0
            )
            self.assertFalse(torch.equal(trunk.weight, model_before[0]))
            self.assertFalse(torch.equal(embedder.weight, model_before[2]))

    def test_subset_weight_mapping_and_empty_mining(self):
        subset = torch.tensor([4, 1, 6, 3])
        labels = torch.tensor([0, 0, 1, 1])
        triplets = (torch.tensor([0, 0]), torch.tensor([1, 1]), torch.tensor([2, 3]))
        weights = subset_weights(8, subset, labels, triplets)
        torch.testing.assert_close(
            weights, torch.tensor([0.0, 1, 0, 0.5, 1, 0, 0.5, 0])
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            weights.tolist(), 32, generator=torch.Generator().manual_seed(17)
        )
        self.assertTrue(set(sampler).issubset(set(subset.tolist())))
        no_index = torch.empty(0, dtype=torch.long)
        empty = (no_index, no_index, no_index)
        torch.testing.assert_close(
            subset_weights(8, subset, labels, empty),
            torch.tensor([0.0, 1, 0, 1, 1, 0, 1, 0]),
        )

    def test_memory_warmup_and_queue_fill(self):
        torch.manual_seed(29)
        model = nn.Linear(4, 2)
        data = torch.randn(6, 4)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        memory = losses.CrossBatchMemory(
            losses.ContrastiveLoss(neg_margin=1.4), 2, memory_size=8
        )
        train_with_memory(model, memory, data, labels, warmup_steps=2, steps=2)
        self.assertEqual(memory.queue_idx, 0)
        self.assertFalse(memory.has_been_filled)
        train_with_memory(model, memory, data, labels, warmup_steps=0, steps=2)
        self.assertEqual(memory.queue_idx, 4)
        self.assertTrue(memory.has_been_filled)
        self.assertFalse(memory.embedding_memory.requires_grad)
        torch.testing.assert_close(
            memory.label_memory, torch.tensor([1, 1, 2, 2, 2, 2, 0, 0])
        )


if __name__ == "__main__":
    unittest.main()


class TestProxyNCAPlusPlusRecipe(unittest.TestCase):
    def test_global_k_max_pool_follows_equation_7(self):
        features = torch.tensor(
            [[[[1.0, 9.0], [3.0, 4.0]], [[8.0, 2.0], [7.0, 0.0]]]],
            device=TEST_DEVICE,
        )
        for k, expected in (
            (1, [9.0, 8.0]),
            (2, [6.5, 7.5]),
            (4, [17 / 4, 17 / 4]),
        ):
            with self.subTest(k=k):
                torch.testing.assert_close(
                    global_k_max_pool(features, k),
                    torch.tensor([expected], device=TEST_DEVICE),
                )

    def test_pool_limits_match_max_and_average_pooling(self):
        features = torch.randn(3, 5, 4, 4, device=TEST_DEVICE)
        torch.testing.assert_close(
            global_k_max_pool(features, 1), features.amax(dim=(2, 3))
        )
        torch.testing.assert_close(
            global_k_max_pool(features, 16), features.mean(dim=(2, 3))
        )

    def test_layer_norm_has_no_parameters(self):
        model = PoolAndEmbed(channels=5, embedding_size=4).to(TEST_DEVICE)
        self.assertEqual(list(model.norm.parameters()), [])
        output = model(torch.randn(3, 5, 2, 2, device=TEST_DEVICE))
        torch.testing.assert_close(
            output.mean(dim=1), torch.zeros(3, device=TEST_DEVICE), atol=1e-6, rtol=0
        )

    def test_proxies_use_their_own_learning_rate(self):
        model = PoolAndEmbed(channels=5, embedding_size=4).to(TEST_DEVICE)
        loss_func = losses.ProxyNCALoss(3, 4, softmax_scale=9).to(TEST_DEVICE)
        optimizer = make_optimizer(model, loss_func, lr=0.01, proxy_lr=2.0)
        data = torch.randn(6, 5, 2, 2, device=TEST_DEVICE)
        labels = torch.arange(3, device=TEST_DEVICE).repeat(2)
        before = loss_func.weight.detach().clone()
        linear_before = model.linear.weight.detach().clone()
        loss_func(model(data), labels).backward()
        assert loss_func.weight.grad is not None
        assert model.linear.weight.grad is not None
        proxy_gradient = loss_func.weight.grad.clone()
        linear_gradient = model.linear.weight.grad.clone()
        optimizer.step()
        torch.testing.assert_close(loss_func.weight, before - 2.0 * proxy_gradient)
        torch.testing.assert_close(
            model.linear.weight, linear_before - 0.01 * linear_gradient
        )

    def test_training_reduces_the_objective(self):
        torch.manual_seed(17)
        labels = torch.arange(4, device=TEST_DEVICE).repeat_interleave(4)
        centers = torch.randn(4, 6, 1, 1, device=TEST_DEVICE) * 3
        data = centers[labels] + 0.3 * torch.randn(16, 6, 3, 3, device=TEST_DEVICE)
        model = PoolAndEmbed(channels=6, embedding_size=4).to(TEST_DEVICE)
        loss_func = losses.ProxyNCALoss(4, 4, softmax_scale=9).to(TEST_DEVICE)
        optimizer = make_optimizer(model, loss_func, lr=0.1, proxy_lr=1.0)
        history = train(model, loss_func, optimizer, data, labels)
        self.assertLess(history[-1], history[0])


def brute_force_violations(embeddings, labels):
    unit = nn.functional.normalize(embeddings, dim=1)
    similarity = (unit @ unit.T).tolist()
    values = labels.tolist()
    classes = sorted(set(values))
    rows = range(len(values))
    table = []
    for selected in classes:
        row = []
        for candidate in classes:
            best = -math.inf
            for a in rows:
                if values[a] != selected:
                    continue
                positives = [
                    similarity[a][p] for p in rows if p != a and values[p] == selected
                ]
                hardest = min(positives) if positives else 0.0
                for n in rows:
                    if values[n] == candidate:
                        best = max(best, similarity[a][n] - hardest)
            row.append(best)
        table.append(row)
    return classes, table


class TestNPairsClassMiningRecipe(unittest.TestCase):
    def setUp(self):
        self.distance = distances.DotProductSimilarity(normalize_embeddings=True)

    def test_violations_match_brute_force(self):
        generator = torch.Generator().manual_seed(1)
        for per_class in (1, 2, 3):
            with self.subTest(per_class=per_class):
                labels = torch.arange(5).repeat_interleave(per_class).to(TEST_DEVICE)
                embeddings = torch.randn(
                    len(labels), 4, generator=generator, dtype=torch.float64
                ).to(TEST_DEVICE)
                classes, table = class_violations(embeddings, labels, self.distance)
                expected_classes, expected = brute_force_violations(embeddings, labels)
                self.assertEqual(classes.tolist(), expected_classes)
                torch.testing.assert_close(
                    table, torch.tensor(expected, dtype=torch.float64).to(TEST_DEVICE)
                )

    def test_distance_scores_are_negated(self):
        generator = torch.Generator().manual_seed(1)
        embeddings = torch.randn(6, 3, generator=generator).to(TEST_DEVICE)
        labels = torch.arange(3).repeat_interleave(2).to(TEST_DEVICE)
        distance = distances.LpDistance()
        _, table = class_violations(embeddings, labels, distance)
        negated = -distance(embeddings)
        expected = negated[0, 3] - negated[0, 1]
        for a, p in ((0, 1), (1, 0)):
            for n in (2, 3):
                expected = torch.maximum(expected, negated[a, n] - negated[a, p])
        self.assertAlmostEqual(table[0, 1].item(), expected.item(), places=5)

    def test_greedy_selection_adds_the_most_violating_class(self):
        angles = [0, 5, 10, 15, 60, 65, 170, 175]
        embeddings = torch.tensor(
            [[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in angles],
            device=TEST_DEVICE,
        )
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=TEST_DEVICE)
        for seed in range(30):
            generator = torch.Generator().manual_seed(seed)
            selected = select_classes(
                embeddings, labels, 2, self.distance, generator
            ).tolist()
            if selected[0] == 0:
                self.assertEqual(selected, [0, 1])
            elif selected[0] == 1:
                self.assertEqual(selected, [1, 0])
            elif selected[0] == 2:
                self.assertEqual(selected[1], 1)

    def test_score_uses_every_selected_class(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=TEST_DEVICE)
        generator = torch.Generator().manual_seed(4)
        embeddings = torch.randn(8, 3, generator=generator).to(TEST_DEVICE)
        _, table = class_violations(embeddings, labels, self.distance)
        for seed in range(40):
            selected = select_classes(
                embeddings,
                labels,
                3,
                self.distance,
                torch.Generator().manual_seed(seed),
            ).tolist()
            remaining = [c for c in range(4) if c not in selected[:2]]
            scores = [table[selected[:2], c].max().item() for c in remaining]
            self.assertEqual(selected[2], remaining[scores.index(max(scores))])

    def test_ties_are_broken_at_random(self):
        embeddings = torch.eye(4, device=TEST_DEVICE).repeat_interleave(2, dim=0)
        labels = torch.arange(4, device=TEST_DEVICE).repeat_interleave(2)
        seen = set()
        for seed in range(60):
            generator = torch.Generator().manual_seed(seed)
            selected = select_classes(embeddings, labels, 2, self.distance, generator)
            if selected[0] == 0:
                seen.add(selected[1].item())
        self.assertEqual(seen, {1, 2, 3})

    def test_first_class_is_random(self):
        generator = torch.Generator().manual_seed(6)
        embeddings = torch.randn(8, 3, generator=generator).to(TEST_DEVICE)
        labels = torch.arange(4, device=TEST_DEVICE).repeat_interleave(2)
        first = {
            select_classes(
                embeddings,
                labels,
                1,
                self.distance,
                torch.Generator().manual_seed(seed),
            ).item()
            for seed in range(40)
        }
        self.assertEqual(first, {0, 1, 2, 3})

    def test_classes_are_never_repeated(self):
        generator = torch.Generator().manual_seed(2)
        embeddings = torch.randn(20, 4, generator=generator).to(TEST_DEVICE)
        labels = torch.arange(5).repeat_interleave(4).to(TEST_DEVICE)
        selected = select_classes(embeddings, labels, 5, self.distance, generator)
        self.assertEqual(sorted(selected.tolist()), [0, 1, 2, 3, 4])

    def test_candidates_are_random_classes_with_two_examples(self):
        labels = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3], device=TEST_DEVICE)
        seen = set()
        for seed in range(30):
            rows = candidate_examples(
                labels, 2, generator=torch.Generator().manual_seed(seed)
            )
            chosen = labels[rows]
            self.assertEqual(len(chosen.unique()), 2)
            self.assertEqual(len(rows.unique()), len(rows))
            self.assertNotIn(2, chosen.tolist())
            seen.update(chosen.tolist())
        self.assertEqual(seen, {0, 1, 3})

    def test_pairs_are_two_examples_of_each_class(self):
        labels = torch.arange(3).repeat_interleave(4).to(TEST_DEVICE)
        classes = torch.tensor([2, 0], device=TEST_DEVICE)
        rows = draw_pairs(labels, classes, torch.Generator().manual_seed(0))
        self.assertEqual(labels[rows].tolist(), [2, 2, 0, 0])
        self.assertEqual(len(rows.unique()), 4)

    def test_npairs_loss_uses_every_mined_pair(self):
        labels = torch.arange(8).repeat_interleave(3).to(TEST_DEVICE)
        generator = torch.Generator().manual_seed(3)
        data = torch.randn(24, 16, generator=generator).to(TEST_DEVICE)
        model = nn.Linear(16, 4).to(TEST_DEVICE)
        loss_func = losses.NPairsLoss()
        batch = mined_batch(model, data, labels, 6, 3, loss_func.distance, generator)
        embeddings = model(data[batch])
        terms = loss_func.loss_terms(embeddings, labels[batch])
        self.assertEqual(len(batch), 6)
        self.assertEqual(len(terms["loss"]["losses"]), 3)

    def test_mined_batches_are_harder_than_random_batches(self):
        torch.manual_seed(5)
        labels = torch.arange(24).repeat_interleave(4).to(TEST_DEVICE)
        centers = nn.functional.normalize(torch.randn(24, 3, device=TEST_DEVICE))
        data = centers[labels] + 0.05 * torch.randn(96, 3, device=TEST_DEVICE)
        model = nn.Identity()
        loss_func = losses.NPairsLoss()
        mined, random_batches = [], []
        for seed in range(40):
            generator = torch.Generator().manual_seed(seed)
            batch = mined_batch(
                model, data, labels, 24, 6, loss_func.distance, generator
            )
            mined.append(loss_func(data[batch], labels[batch]).item())
            classes = torch.randperm(24, generator=generator)[:6].to(TEST_DEVICE)
            batch = draw_pairs(labels, classes, generator)
            random_batches.append(loss_func(data[batch], labels[batch]).item())
        self.assertGreater(sum(mined) / 40, sum(random_batches) / 40 + 0.2)

    def test_inference_builds_no_graph_and_keeps_the_model_mode(self):
        labels = torch.arange(4).repeat_interleave(2).to(TEST_DEVICE)
        data = torch.randn(8, 4, device=TEST_DEVICE)
        model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5)).to(TEST_DEVICE)
        model.train()
        gradient_modes = []
        model.register_forward_hook(
            lambda *_: gradient_modes.append(torch.is_grad_enabled())
        )
        mined_batch(model, data, labels, 4, 2, losses.NPairsLoss().distance)
        self.assertTrue(model.training)
        self.assertEqual(gradient_modes, [False])

    def test_invalid_requests(self):
        labels = torch.tensor([0, 0, 1, 1, 2], device=TEST_DEVICE)
        embeddings = torch.randn(5, 2, device=TEST_DEVICE)
        with self.assertRaisesRegex(ValueError, "num_classes"):
            candidate_examples(labels, 3)
        for count in (0, 4):
            with (
                self.subTest(count=count),
                self.assertRaisesRegex(ValueError, "num_pairs"),
            ):
                select_classes(embeddings, labels, count, self.distance)
        with self.assertRaisesRegex(ValueError, "two examples"):
            draw_pairs(labels, torch.tensor([2], device=TEST_DEVICE))
