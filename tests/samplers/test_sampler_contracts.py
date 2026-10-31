import unittest
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_metric_learning.samplers import (
    FixedSetOfPairs,
    FixedSetOfTriplets,
    HierarchicalSampler,
    MPerClassSampler,
)


def hierarchical_labels():
    classes = torch.arange(24).repeat_interleave(3)
    return torch.stack([classes, classes // 6], dim=1)


def make_samplers(generator):
    labels = torch.arange(10).repeat_interleave(4)
    return [
        MPerClassSampler(labels, 2, batch_size=8, num_batches=8, generator=generator),
        FixedSetOfPairs(labels, 20, batch_size=8, generator=generator),
        FixedSetOfTriplets(labels, 20, generator=generator),
        HierarchicalSampler(hierarchical_labels(), 12, 3, generator=generator),
    ]


class TestSamplerContracts(unittest.TestCase):
    def test_samplers_reject_impossible_batches(self):
        with self.assertRaisesRegex(ValueError, "m must be positive"):
            MPerClassSampler([0, 0, 1, 1], m=0, batch_size=2)
        with self.assertRaisesRegex(ValueError, "labels must not be empty"):
            MPerClassSampler([], m=2, batch_size=2)
        with self.assertRaisesRegex(ValueError, "at least two classes"):
            FixedSetOfTriplets([0, 0], num_triplets=2)
        for num_triplets in (0, -1):
            with self.assertRaisesRegex(ValueError, "num_triplets"):
                FixedSetOfTriplets([0, 0, 1, 1], num_triplets=num_triplets)

    def test_hierarchical_sampler_rejects_invalid_batches(self):
        labels = hierarchical_labels()
        cases: list[tuple[dict[str, Any], str]] = [
            ({"batch_size": 0, "samples_per_class": 3}, "must be positive"),
            (
                {
                    "batch_size": 12,
                    "samples_per_class": 3,
                    "super_classes_per_batch": 0,
                },
                "must be positive",
            ),
            (
                {
                    "batch_size": 12,
                    "samples_per_class": 3,
                    "batches_per_super_tuple": -1,
                },
                "nonnegative",
            ),
            (
                {"batch_size": 13, "samples_per_class": 3},
                "multiple of 2",
            ),
            ({"batch_size": 12, "samples_per_class": 0}, "positive or"),
            ({"batch_size": 12, "samples_per_class": 4}, "samples_per_class"),
        ]
        for arguments, message in cases:
            with self.subTest(**arguments), self.assertRaisesRegex(ValueError, message):
                HierarchicalSampler(labels, **arguments)
        sampler = HierarchicalSampler(labels, batch_size=12, samples_per_class="all")
        self.assertTrue(all(len(batch) <= 12 for batch in sampler))

    def test_explicit_generator_reproduces_every_iteration(self):
        first = make_samplers(torch.Generator().manual_seed(3))
        second = make_samplers(torch.Generator().manual_seed(3))
        for a, b in zip(first, second, strict=True):
            with self.subTest(sampler=type(a).__name__):
                epochs = [list(a) for _ in range(2)]
                self.assertEqual(epochs, [list(b) for _ in range(2)])
                self.assertNotEqual(epochs[0], epochs[1])

    def test_global_seed_controls_default_generator(self):
        torch.manual_seed(11)
        first = [list(s) for s in make_samplers(None)]
        torch.manual_seed(11)
        second = [list(s) for s in make_samplers(None)]
        self.assertEqual(first, second)

    def test_length_matches_yielded_items(self):
        for sampler in make_samplers(None):
            with self.subTest(sampler=type(sampler).__name__):
                items = list(sampler)
                self.assertEqual(len(items), len(sampler))

    def test_m_per_class_sampler_is_a_batch_sampler(self):
        labels = torch.arange(10).repeat_interleave(4)
        sampler = MPerClassSampler(labels, 2, batch_size=8, num_batches=8)
        loader = DataLoader(TensorDataset(labels), batch_sampler=sampler)
        batches = [batch_labels for (batch_labels,) in loader]
        self.assertEqual(len(batches), 8)
        for batch_labels in batches:
            _, counts = torch.unique(batch_labels, return_counts=True)
            self.assertEqual(counts.tolist(), [2, 2, 2, 2])

    def test_hierarchical_sampler_is_a_batch_sampler(self):
        labels = hierarchical_labels()
        sampler = HierarchicalSampler(labels, 12, 3)
        loader = DataLoader(TensorDataset(labels), batch_sampler=sampler)
        batches = [batch for (batch,) in loader]
        self.assertEqual(len(batches), len(sampler))
        for batch in batches:
            self.assertEqual(batch.shape, (12, 2))
            _, super_counts = torch.unique(batch[:, 1], return_counts=True)
            self.assertEqual(super_counts.tolist(), [6, 6])
            _, class_counts = torch.unique(batch[:, 0], return_counts=True)
            self.assertTrue(torch.all(class_counts == 3))


if __name__ == "__main__":
    unittest.main()
