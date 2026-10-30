import unittest
from collections import Counter

import torch

from pytorch_metric_learning.samplers import MPerClassSampler


def class_counts(batch, labels):
    return Counter(labels[index] for index in batch)


class TestMPerClassSampler(unittest.TestCase):
    def test_every_batch_has_m_samples_of_distinct_classes(self):
        labels = torch.randint(0, 20, (500,)).tolist()
        num_classes = len(set(labels))
        for batch_size, m in ((4, 2), (40, 2), (20, 5), (17, 17), (6, 1)):
            with self.subTest(batch_size=batch_size, m=m):
                sampler = MPerClassSampler(labels, m, batch_size, num_batches=50)
                batches = list(sampler)
                self.assertEqual(len(batches), 50)
                self.assertLessEqual(batch_size // m, num_classes)
                for batch in batches:
                    self.assertEqual(len(batch), batch_size)
                    self.assertEqual(set(class_counts(batch, labels).values()), {m})
                    self.assertEqual(len(class_counts(batch, labels)), batch_size // m)

    def test_rejects_impossible_batches(self):
        labels = [0, 0, 1, 1, 2, 2]
        cases = (
            ({"m": 2, "batch_size": 8}, "unique labels"),
            ({"m": 2, "batch_size": 5}, "without any remainder"),
            ({"m": 2, "batch_size": 0}, "batch_size must be positive"),
            ({"m": 2, "batch_size": 4, "num_batches": 0}, "num_batches"),
        )
        for arguments, message in cases:
            with self.subTest(**arguments), self.assertRaisesRegex(ValueError, message):
                MPerClassSampler(labels, **arguments)

    def test_default_length_visits_each_example_exactly_once_when_divisible(self):
        labels = torch.arange(12).repeat_interleave(6).tolist()
        sampler = MPerClassSampler(labels, 3, batch_size=12)
        self.assertEqual(len(sampler), 6)
        for _ in range(5):
            indices = [index for batch in sampler for index in batch]
            self.assertEqual(sorted(indices), list(range(len(labels))))

    def test_default_length_visits_each_example_with_unequal_classes(self):
        sizes = [1, 2, 3, 5, 8, 13, 21]
        labels = [c for c, size in enumerate(sizes) for _ in range(size)]
        for m, batch_size, expected_length in (
            (2, 6, 26),
            (4, 12, 14),
            (3, 21, 7),
        ):
            with self.subTest(m=m, batch_size=batch_size):
                sampler = MPerClassSampler(labels, m, batch_size)
                self.assertEqual(len(sampler), expected_length)
                for _ in range(20):
                    seen = {index for batch in sampler for index in batch}
                    self.assertEqual(seen, set(range(len(labels))))

    def test_default_length_is_minimal_for_the_largest_class(self):
        labels = [0] * 9 + [1] * 2
        sampler = MPerClassSampler(labels, 3, batch_size=3)
        self.assertEqual(len(sampler), 6)
        shorter = MPerClassSampler(labels, 3, batch_size=3, num_batches=5)
        seen = [{index for batch in shorter for index in batch} for _ in range(50)]
        self.assertLess(min(map(len, seen)), len(labels))

    def test_counts_stay_within_one_at_every_batch(self):
        sizes = [2, 3, 7, 11, 30]
        labels = [c for c, size in enumerate(sizes) for _ in range(size)]
        sampler = MPerClassSampler(labels, 2, batch_size=6, num_batches=101)
        classes, examples = Counter(), Counter()
        for batch in sampler:
            classes.update({labels[index] for index in batch})
            examples.update(batch)
            self.assertLessEqual(
                max(classes[c] for c in range(5)) - min(classes[c] for c in range(5)),
                1,
            )
            for c in range(5):
                members = [examples[i] for i, y in enumerate(labels) if y == c]
                self.assertLessEqual(max(members) - min(members), 1)

    def test_batches_do_not_repeat_examples_of_large_enough_classes(self):
        labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        sampler = MPerClassSampler(labels, 2, batch_size=4, num_batches=500)
        for batch in sampler:
            self.assertEqual(len(set(batch)), 4)

    def test_small_classes_repeat_examples_evenly_within_a_batch(self):
        labels = [0, 1, 1, 1, 1, 2, 2]
        sampler = MPerClassSampler(labels, 3, batch_size=9, num_batches=20)
        for batch in sampler:
            self.assertEqual(batch.count(0), 3)
            self.assertEqual(len({i for i in batch if labels[i] == 1}), 3)
            twos = sorted(batch.count(i) for i in (5, 6))
            self.assertEqual(twos, [1, 2])

    def test_iterators_are_independent_draws(self):
        generator = torch.Generator().manual_seed(0)
        labels = torch.arange(8).repeat_interleave(8)
        sampler = MPerClassSampler(labels, 4, batch_size=16, generator=generator)
        self.assertNotEqual(list(sampler), list(sampler))

    def test_accepts_hashable_labels(self):
        labels = ["cat", "dog", "cat", "dog"]
        batches = list(MPerClassSampler(labels, 2, batch_size=4))
        self.assertEqual(sorted(batches[0]), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
