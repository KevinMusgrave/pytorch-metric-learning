import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_metric_learning import losses
from pytorch_metric_learning.samplers import FixedSetOfPairs

from .. import TEST_DEVICE


def seeded(seed=0):
    return torch.Generator().manual_seed(seed)


class TestFixedSetOfPairs(unittest.TestCase):
    def test_pairs_are_balanced_and_valid(self):
        labels = torch.arange(10).repeat_interleave(3)
        for num_pairs, positives in ((1, 1), (7, 4), (20, 10), (301, 151)):
            sampler = FixedSetOfPairs(labels, num_pairs, 8, generator=seeded())
            first, second = labels[sampler.fixed_set_of_pairs].unbind(1)
            self.assertEqual(int((first == second).sum()), positives)
            self.assertEqual(int((first != second).sum()), num_pairs - positives)
            self.assertFalse(
                (
                    sampler.fixed_set_of_pairs[:, 0] == sampler.fixed_set_of_pairs[:, 1]
                ).any()
            )

    def test_every_iterator_yields_the_same_pairs_in_new_order(self):
        labels = torch.arange(6).repeat_interleave(4)
        sampler = FixedSetOfPairs(labels, 21, 6, generator=seeded())
        expected = sorted(map(tuple, sampler.fixed_set_of_pairs.tolist()))
        orders = []
        for _ in range(3):
            batches = list(sampler)
            self.assertEqual(len(batches), len(sampler))
            self.assertEqual([len(batch) for batch in batches], [6] * 7)
            flat = torch.tensor(batches).reshape(-1, 2)
            self.assertEqual(sorted(map(tuple, flat.tolist())), expected)
            orders.append(flat.tolist())
        self.assertNotEqual(orders[0], orders[1])

    def test_short_final_batch_keeps_whole_pairs(self):
        labels = torch.arange(4).repeat_interleave(4)
        sampler = FixedSetOfPairs(labels, 7, 6, generator=seeded())
        self.assertEqual(len(sampler), 3)
        self.assertEqual([len(batch) for batch in sampler], [6, 6, 2])

    def test_singleton_classes_join_only_negative_pairs(self):
        labels = [0, 0, 1, 2, 3]
        sampler = FixedSetOfPairs(labels, 40, 4, generator=seeded())
        pair_labels = torch.tensor(labels)[sampler.fixed_set_of_pairs]
        positives = pair_labels[pair_labels[:, 0] == pair_labels[:, 1]]
        self.assertEqual(positives.unique().tolist(), [0])
        self.assertEqual(len(positives), 20)
        self.assertEqual(len(pair_labels.unique()), 4)

    def test_hashable_labels(self):
        labels = ["cat", "cat", "dog", "dog"]
        sampler = FixedSetOfPairs(labels, 4, 2, generator=seeded())
        self.assertEqual(len(sampler), 4)

    def test_same_seed_repeats_pairs_and_order(self):
        labels = torch.arange(5).repeat_interleave(4)
        runs = [
            list(FixedSetOfPairs(labels, 12, 4, generator=seeded(3))) for _ in range(2)
        ]
        self.assertEqual(runs[0], runs[1])

    def test_invalid_arguments(self):
        cases = [
            ([0, 0, 1, 1], 0, 2, "num_pairs"),
            ([0, 0, 1, 1], -1, 2, "num_pairs"),
            ([0, 0, 1, 1], 2, 0, "batch_size"),
            ([0, 0, 1, 1], 2, 3, "batch_size"),
            ([0, 0, 0], 2, 2, "two classes"),
            ([], 2, 2, "two classes"),
            ([0, 1, 2], 2, 2, "two examples"),
        ]
        for labels, num_pairs, batch_size, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                FixedSetOfPairs(labels, num_pairs, batch_size)

    def test_dataloader_to_loss_uses_exactly_the_fixed_pairs(self):
        labels = torch.arange(8).repeat_interleave(4)
        dataset = TensorDataset(
            torch.arange(len(labels), dtype=torch.float32).unsqueeze(1), labels
        )
        sampler = FixedSetOfPairs(labels, 24, 8, generator=seeded())
        loader = DataLoader(dataset, batch_sampler=sampler)
        expected = sorted(map(tuple, sampler.fixed_set_of_pairs.tolist()))
        seen = []
        generator = torch.Generator().manual_seed(1)
        features = torch.randn(len(labels), 6, generator=generator).to(TEST_DEVICE)
        model = torch.nn.Linear(6, 3).to(TEST_DEVICE)
        loss_func = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        for rows, batch_labels in loader:
            indices = rows.long().flatten().to(TEST_DEVICE)
            batch_labels = batch_labels.to(TEST_DEVICE)
            seen.extend(indices.reshape(-1, 2).tolist())
            embeddings = model(features[indices])
            first, second = torch.arange(len(embeddings)).reshape(-1, 2).unbind(1)
            first, second = first.to(TEST_DEVICE), second.to(TEST_DEVICE)
            positive = batch_labels[first] == batch_labels[second]
            loss = loss_func(
                embeddings,
                batch_labels,
                indices_tuple=(
                    first[positive],
                    second[positive],
                    first[~positive],
                    second[~positive],
                ),
            )
            normalized = torch.nn.functional.normalize(embeddings.detach(), dim=1)
            distances = (normalized[first] - normalized[second]).norm(dim=1)
            pos_terms = distances[positive]
            neg_terms = (1 - distances[~positive]).clamp_min(0)
            pos_mean = pos_terms[pos_terms > 0].mean() if (pos_terms > 0).any() else 0
            neg_mean = neg_terms[neg_terms > 0].mean() if (neg_terms > 0).any() else 0
            self.assertAlmostEqual(
                loss.item(), float(pos_mean) + float(neg_mean), places=5
            )
        self.assertEqual(sorted(map(tuple, seen)), expected)
