import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from pytorch_metric_learning.samplers import (
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
        MPerClassSampler(
            labels, 2, batch_size=8, length_before_new_iter=64, generator=generator
        ),
        FixedSetOfTriplets(labels, 20, generator=generator),
        HierarchicalSampler(hierarchical_labels(), 12, 3, generator=generator),
    ]


class TestSamplerContracts(unittest.TestCase):
    def test_samplers_reject_impossible_batches(self):
        with self.assertRaisesRegex(ValueError, "m must be positive"):
            MPerClassSampler([0, 0, 1, 1], m=0)
        with self.assertRaisesRegex(ValueError, "labels must not be empty"):
            MPerClassSampler([], m=2)
        with self.assertRaisesRegex(ValueError, "at least two classes"):
            FixedSetOfTriplets([0, 0], num_triplets=2)

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
        labels = torch.arange(100).repeat(3)
        for batch_size, length_before_new_iter in (
            (None, 100),
            (None, 1999),
            (20, 100),
        ):
            with self.subTest(
                batch_size=batch_size, length_before_new_iter=length_before_new_iter
            ):
                sampler = MPerClassSampler(
                    labels,
                    5,
                    batch_size=batch_size,
                    length_before_new_iter=length_before_new_iter,
                )
                indices = list(sampler)
                self.assertEqual(len(indices), len(sampler))
                self.assertTrue(all(0 <= i < len(labels) for i in indices))
        for sampler in make_samplers(None)[1:]:
            with self.subTest(sampler=type(sampler).__name__):
                items = list(sampler)
                self.assertEqual(len(items), len(sampler))

    def test_m_per_class_batches_in_a_dataloader(self):
        labels = torch.arange(10).repeat_interleave(4)
        sampler = MPerClassSampler(labels, 2, batch_size=8, length_before_new_iter=64)
        loader = DataLoader(TensorDataset(labels), batch_size=8, sampler=sampler)
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

    def test_small_classes_are_sampled_with_replacement(self):
        labels = torch.tensor([0, 1, 1, 1])
        sampler = MPerClassSampler(labels, 3, batch_size=6, length_before_new_iter=6)
        indices = list(sampler)
        self.assertEqual(sorted(labels[indices].tolist()), [0, 0, 0, 1, 1, 1])
        self.assertEqual(sorted(i for i in indices if labels[i] == 1), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
