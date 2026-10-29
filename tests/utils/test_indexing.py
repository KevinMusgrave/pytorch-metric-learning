import unittest

import torch

from pytorch_metric_learning.types import TripletIndices
from pytorch_metric_learning.utils import _common as c_f
from pytorch_metric_learning.utils import indexing

from .. import TEST_DEVICE, TEST_DTYPES, helpers


class TestLossAndMinerUtils(unittest.TestCase):
    def test_logsumexp(self):
        for dtype in TEST_DTYPES:
            rtol = helpers.rtol(dtype)
            mat = torch.tensor(
                [
                    [-1, 0, 1, 10, 50],
                    [-300, -200, -100, -50, -20],
                    [-300, -200, 0, 200, 300],
                    [100, 200, 300, 400, 500],
                    [0, 0, 0, 0, 0],
                ],
                dtype=dtype,
                requires_grad=True,
            ).to(TEST_DEVICE)
            result = c_f.logsumexp(mat, keep_mask=None, add_one=False, dim=1)
            torch.mean(result).backward(retain_graph=True)
            correct_result = torch.logsumexp(mat, dim=1, keepdim=True)
            self.assertTrue(torch.allclose(result, correct_result, rtol=rtol))

            result = c_f.logsumexp(mat, keep_mask=None, add_one=True, dim=1)
            torch.mean(result).backward(retain_graph=True)
            correct_result = torch.logsumexp(
                torch.cat(
                    [
                        mat,
                        torch.zeros(mat.size(0), dtype=dtype)
                        .to(TEST_DEVICE)
                        .unsqueeze(1),
                    ],
                    dim=1,
                ),
                dim=1,
                keepdim=True,
            )
            self.assertTrue(torch.allclose(result, correct_result, rtol=rtol))

            keep_mask = torch.tensor(
                [
                    [1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                ],
                dtype=torch.bool,
            ).to(TEST_DEVICE)
            result = c_f.logsumexp(mat, keep_mask=keep_mask, add_one=False, dim=1)
            torch.mean(result).backward()

            row0_input = torch.tensor([-1, 0], dtype=dtype).to(TEST_DEVICE)
            row1_input = torch.tensor([-300, -200, -100, -50, -20], dtype=dtype).to(
                TEST_DEVICE
            )
            row2_input = torch.tensor([-200, 0, 200], dtype=dtype).to(TEST_DEVICE)
            row4_input = torch.tensor([0, 0], dtype=dtype).to(TEST_DEVICE)

            row0 = torch.logsumexp(row0_input, dim=0).unsqueeze(0)
            row1 = torch.logsumexp(row1_input, dim=0).unsqueeze(0)
            row2 = torch.logsumexp(row2_input, dim=0).unsqueeze(0)
            row3 = torch.tensor([0.0], dtype=dtype).to(TEST_DEVICE)
            row4 = torch.logsumexp(row4_input, dim=0).unsqueeze(0)
            correct_result = torch.stack([row0, row1, row2, row3, row4], dim=0)
            self.assertTrue(torch.allclose(result, correct_result, rtol=rtol))

    def test_get_all_pairs_triplets_indices(self):
        original_x = torch.arange(10)

        for i in range(1, 11):
            x = original_x.repeat(i)
            correct_num_pos = len(x) * (i - 1)
            correct_num_neg = len(x) * (len(x) - i)
            a1, p, a2, n = indexing.get_all_pairs_indices(x)
            self.assertTrue(len(a1) == len(p) == correct_num_pos)
            self.assertTrue(len(a2) == len(n) == correct_num_neg)

            correct_num_triplets = len(x) * (i - 1) * (len(x) - i)
            a, p, n = indexing.get_all_triplets_indices(x)
            self.assertTrue(len(a) == len(p) == len(n) == correct_num_triplets)

    def test_convert_to_triplets(self):
        a1 = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        p = torch.tensor([4, 4, 4, 4], dtype=torch.long)
        a2 = torch.tensor([4, 5, 6, 7], dtype=torch.long)
        n = torch.tensor([5, 5, 6, 6], dtype=torch.long)
        triplets = indexing.convert_to_triplets((a1, p, a2, n), labels=torch.arange(7))
        self.assertTrue(all(len(x) == 0 for x in triplets))

        a2 = torch.tensor([0, 4, 5, 6], dtype=torch.long)
        triplets = indexing.convert_to_triplets((a1, p, a2, n), labels=torch.arange(7))
        self.assertTrue(
            triplets == (torch.tensor([0]), torch.tensor([4]), torch.tensor([5]))
        )

        a1 = torch.tensor([0, 1, 0, 2], dtype=torch.long)
        p = torch.tensor([5, 6, 7, 8], dtype=torch.long)
        a2 = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        n = torch.tensor([9, 10, 11, 12], dtype=torch.long)
        triplets = indexing.convert_to_triplets((a1, p, a2, n), labels=torch.arange(13))
        triplets = torch.stack(triplets, dim=1)
        found_set = set()
        for t in triplets:
            found_set.add(tuple(t.cpu().numpy()))
        correct_triplets = {
            (0, 5, 9),
            (0, 5, 12),
            (0, 7, 9),
            (0, 7, 12),
            (1, 6, 10),
            (2, 8, 11),
        }

        self.assertTrue(found_set == correct_triplets)

    def test_convert_to_weights(self):
        a = torch.tensor([0, 1, 2, 3], dtype=torch.long).to(TEST_DEVICE)
        p = torch.tensor([4, 4, 4, 4], dtype=torch.long).to(TEST_DEVICE)
        n = torch.tensor([5, 5, 6, 6], dtype=torch.long).to(TEST_DEVICE)
        for dtype in TEST_DTYPES:
            weights = indexing.convert_to_weights(
                TripletIndices(a, p, n),
                labels=torch.arange(7).to(TEST_DEVICE),
                dtype=dtype,
            )
            correct_weights = torch.tensor(
                [0.25, 0.25, 0.25, 0.25, 1, 0.5, 0.5], dtype=dtype
            ).to(TEST_DEVICE)
            self.assertTrue(torch.all(weights == correct_weights))

        a = torch.tensor([], dtype=torch.long).to(TEST_DEVICE)
        p = torch.tensor([], dtype=torch.long).to(TEST_DEVICE)
        n = torch.tensor([], dtype=torch.long).to(TEST_DEVICE)
        for dtype in TEST_DTYPES:
            weights = indexing.convert_to_weights(
                TripletIndices(a, p, n),
                labels=torch.arange(7).to(TEST_DEVICE),
                dtype=dtype,
            )
            correct_weights = torch.tensor([1, 1, 1, 1, 1, 1, 1], dtype=dtype).to(
                TEST_DEVICE
            )
            self.assertTrue(torch.all(weights == correct_weights))

        batch_size = 32
        labels = torch.randint(0, 10, size=(batch_size,)).to(TEST_DEVICE)
        a = torch.randint(0, batch_size, size=(256,)).to(TEST_DEVICE)
        p = torch.randint(0, batch_size * 2, size=(256,)).to(TEST_DEVICE)
        n = torch.randint(0, batch_size * 2, size=(256,)).to(TEST_DEVICE)
        for dtype in TEST_DTYPES:
            weights = indexing.convert_to_weights(
                TripletIndices(a, p, n), labels=labels, dtype=dtype, using_ref=True
            )

            _, counts = torch.unique(a, return_counts=True, sorted=True)
            counts = counts.type(weights.dtype) / torch.max(counts)

            self.assertTrue(torch.equal(weights, counts))

    def test_get_random_triplet_indices(self):
        for dtype in TEST_DTYPES:
            for _ in range(10):
                labels = (
                    torch.randint(low=0, high=5, size=(100,))
                    .to(TEST_DEVICE)
                    .type(dtype)
                )
                a, p, n = indexing.get_random_triplet_indices(labels)

                l_a, l_p, l_n = labels[a], labels[p], labels[n]
                self.assertTrue(torch.all(a != p))
                self.assertTrue(torch.all(l_a == l_p))
                self.assertTrue(torch.all(l_p != l_n))

                labels = (
                    torch.randint(low=0, high=5, size=(100,))
                    .to(TEST_DEVICE)
                    .type(dtype)
                )
                ref_labels = (
                    torch.randint(low=0, high=5, size=(50,)).to(TEST_DEVICE).type(dtype)
                )
                a, p, n = indexing.get_random_triplet_indices(
                    labels, ref_labels=ref_labels
                )

                l_a = labels[a]
                l_p = ref_labels[p]
                l_n = ref_labels[n]
                self.assertTrue(torch.all(l_a == l_p))
                self.assertTrue(torch.all(l_p != l_n))

                labels = torch.randint(0, 3, (10,)).to(TEST_DEVICE).type(dtype)
                w = torch.zeros((len(labels), len(labels))).to(TEST_DEVICE).type(dtype)
                for i, label in enumerate(labels):
                    ind = torch.where(labels != label)[0]
                    j = ind[torch.randint(0, len(ind), (1,))[0]]
                    w[i, j] = 1.0
                a, p, n = indexing.get_random_triplet_indices(
                    labels, t_per_anchor=1, weights=w
                )
                self.assertTrue(torch.all(w[a, n] == 1.0))

                labels = torch.randint(0, 3, (100,)).to(TEST_DEVICE).type(dtype)
                ref_labels = torch.randint(0, 3, (50,)).to(TEST_DEVICE).type(dtype)
                w = (
                    torch.zeros((len(labels), len(ref_labels)))
                    .to(TEST_DEVICE)
                    .type(dtype)
                )
                for i, label in enumerate(labels):
                    ind = torch.where(ref_labels != label)[0]
                    j = ind[torch.randint(0, len(ind), (1,))[0]]
                    w[i, j] = 1.0
                a, p, n = indexing.get_random_triplet_indices(
                    labels, ref_labels=ref_labels, t_per_anchor=1, weights=w
                )
                self.assertTrue(torch.all(w[a, n] == 1.0))

                labels = torch.arange(100).to(TEST_DEVICE).type(dtype)
                a, p, n = indexing.get_random_triplet_indices(labels)
                self.assertTrue(len(a) == len(p) == len(n) == 0)

                labels = torch.zeros(100).to(TEST_DEVICE).type(dtype)
                a, p, n = indexing.get_random_triplet_indices(labels)
                self.assertTrue(len(a) == len(p) == len(n) == 0)

                labels = torch.zeros(100).to(TEST_DEVICE).type(dtype)
                ref_labels = torch.ones(100).to(TEST_DEVICE).type(dtype)
                a, p, n = indexing.get_random_triplet_indices(
                    labels, ref_labels=ref_labels
                )
                self.assertTrue(len(a) == len(p) == len(n) == 0)

                labels = torch.zeros(100).to(TEST_DEVICE).type(dtype)
                ref_labels = torch.zeros(100).to(TEST_DEVICE).type(dtype)
                a, p, n = indexing.get_random_triplet_indices(
                    labels, ref_labels=ref_labels
                )
                self.assertTrue(len(a) == len(p) == len(n) == 0)

                labels = torch.zeros(100).to(TEST_DEVICE).type(dtype)
                ref_labels = torch.ones(100).to(TEST_DEVICE).type(dtype)
                ref_labels[0] = 0
                a, p, n = indexing.get_random_triplet_indices(
                    labels, ref_labels=ref_labels
                )
                self.assertTrue(len(a) == len(p) == len(n) == 100)

    def test_remove_self_comparisons(self):
        labels = torch.tensor([0, 0, 1])
        ref_labels = torch.tensor([1, 1, 0, 0])

        for do_reverse in [False, True]:
            if do_reverse:
                all_labels = torch.cat([ref_labels, labels])
                curr_batch_idx = torch.arange(4, 7)
                correct_a1 = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])
                correct_p = torch.tensor([2, 3, 5, 2, 3, 4, 0, 1])
            else:
                all_labels = torch.cat([labels, ref_labels])
                curr_batch_idx = torch.arange(3)
                correct_a1 = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])
                correct_p = torch.tensor([1, 5, 6, 0, 5, 6, 3, 4])

            indices_tuple = indexing.get_all_pairs_indices(labels, all_labels)
            indices_tuple = indexing.remove_self_comparisons(
                indices_tuple, curr_batch_idx
            )
            a1, p, _, _ = indices_tuple
            self.assertTrue(torch.equal(a1, correct_a1))
            self.assertTrue(torch.equal(p, correct_p))

    def test_remove_self_comparisons_with_wrapped_queue_slots(self):
        labels = torch.tensor([0, 1, 0])
        memory_labels = torch.tensor([0, 0, 1, 7, 0])
        curr_batch_idx = torch.tensor([3, 4, 0])
        memory_labels[curr_batch_idx] = labels
        a, p, n = indexing.get_all_triplets_indices(labels, memory_labels)
        kept = indexing.remove_self_comparisons(TripletIndices(a, p, n), curr_batch_idx)
        self.assertTrue(torch.all(kept.positives != curr_batch_idx[kept.anchors]))
        removed = set(zip(a.tolist(), p.tolist(), strict=True)) - set(
            zip(kept.anchors.tolist(), kept.positives.tolist(), strict=True)
        )
        self.assertEqual(removed, {(0, 3), (1, 4), (2, 0)})


class TestUniqueLabelPositivePairs(unittest.TestCase):
    def test_first_pair_for_each_anchor_label_in_label_order(self):
        torch.manual_seed(0)
        for _ in range(20):
            labels = torch.randint(0, 6, (30,), device=TEST_DEVICE)
            a, p, _, _ = indexing.get_all_pairs_indices(labels)
            expected = {}
            for anchor, positive in zip(a.tolist(), p.tolist(), strict=True):
                expected.setdefault(labels[anchor].item(), (anchor, positive))
            ordered = [expected[label] for label in sorted(expected)]
            actual_a, actual_p = indexing.convert_to_pos_pairs_with_unique_labels(
                None, labels
            )
            self.assertEqual(
                list(zip(actual_a.tolist(), actual_p.tolist(), strict=True)), ordered
            )


if __name__ == "__main__":
    unittest.main()
