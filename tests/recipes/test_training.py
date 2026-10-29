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
from docs.recipes.training import train_metric, train_proxy, train_two_stream
from pytorch_metric_learning import losses, miners
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
