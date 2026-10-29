import unittest

import torch

from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import (
    ContrastiveLoss,
    CrossBatchMemory,
    MultiSimilarityLoss,
    NTXentLoss,
)
from pytorch_metric_learning.miners import (
    BatchEasyHardMiner,
    DistanceWeightedMiner,
    MultiSimilarityMiner,
    PairMarginMiner,
    TripletMarginMiner,
)
from pytorch_metric_learning.types import PairIndices, TripletIndices
from pytorch_metric_learning.utils import indexing

from .. import TEST_DEVICE, TEST_DTYPES
from ..helpers import angle_to_coord


def recent(all_labels, batch_size):
    return torch.arange(
        len(all_labels) - batch_size, len(all_labels), device=all_labels.device
    )


class TestCrossBatchMemory(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.embedding_size = 128
        self.memory_size = 321

    @classmethod
    def tearDown(self):
        torch.cuda.empty_cache()

    def test_remove_self_comparisons(self):
        for dtype in TEST_DTYPES:
            batch_size = 32
            loss = CrossBatchMemory(
                loss=ContrastiveLoss(),
                embedding_size=self.embedding_size,
                memory_size=self.memory_size,
            ).to(TEST_DEVICE, dtype)
            loss.embedding_memory = loss.embedding_memory.to(TEST_DEVICE).type(dtype)
            loss.label_memory = loss.label_memory.to(TEST_DEVICE)
            num_tuples = 1000
            num_non_identical = 147

            for step in range(30):
                start = step * batch_size
                curr_batch_idx = (
                    torch.arange(start, start + batch_size, device=TEST_DEVICE)
                    % self.memory_size
                )
                for identical in [True, False]:
                    a = torch.randint(0, batch_size, (num_tuples,)).to(TEST_DEVICE)
                    p = a + curr_batch_idx[0]
                    if not identical:
                        rand_diff_idx = torch.randint(
                            0, num_tuples, (num_non_identical,)
                        ).to(TEST_DEVICE)
                        offsets = torch.randint(
                            -self.memory_size + 1,
                            self.memory_size,
                            (num_non_identical,),
                        ).to(TEST_DEVICE)
                        offsets[offsets == 0] = 1
                        p[rand_diff_idx] += offsets
                    p %= self.memory_size
                    n = torch.randint(0, batch_size, (num_tuples,)).to(TEST_DEVICE)
                    a_new, p_new, n_new = indexing.remove_self_comparisons(
                        TripletIndices(a, p, n), curr_batch_idx
                    )
                    if identical:
                        self.assertTrue(len(a_new) == len(p_new) == len(n_new) == 0)
                    else:
                        triplets = {
                            (x.item(), y.item(), z.item())
                            for x, y, z in zip(
                                a[rand_diff_idx],
                                p[rand_diff_idx],
                                n[rand_diff_idx],
                                strict=True,
                            )
                        }
                        triplets_new = {
                            (x.item(), y.item(), z.item())
                            for x, y, z in zip(a_new, p_new, n_new, strict=True)
                        }
                        self.assertTrue(set(triplets) == set(triplets_new))

                    a1 = torch.randint(0, batch_size, (num_tuples,)).to(TEST_DEVICE)
                    p = a1 + curr_batch_idx[0]
                    if not identical:
                        rand_diff_idx = torch.randint(
                            0, num_tuples, (num_non_identical,)
                        ).to(TEST_DEVICE)
                        offsets = torch.randint(
                            -self.memory_size + 1,
                            self.memory_size,
                            (num_non_identical,),
                        ).to(TEST_DEVICE)
                        offsets[offsets == 0] = 1
                        p[rand_diff_idx] += offsets
                    p %= self.memory_size
                    a2 = torch.randint(0, batch_size, (num_tuples,)).to(TEST_DEVICE)
                    n = (
                        torch.randint(0, batch_size, (num_tuples,)).to(TEST_DEVICE)
                        + curr_batch_idx[0]
                    ) % self.memory_size
                    a1_new, p_new, a2_new, n_new = indexing.remove_self_comparisons(
                        PairIndices(a1, p, a2, n), curr_batch_idx
                    )
                    if identical:
                        self.assertTrue(len(a1_new) == len(p_new) == 0)
                    else:
                        pos_pairs = {
                            (x.item(), y.item())
                            for x, y in zip(
                                a1[rand_diff_idx], p[rand_diff_idx], strict=True
                            )
                        }
                        pos_pairs_new = {
                            (x.item(), y.item())
                            for x, y in zip(a1_new, p_new, strict=True)
                        }
                        self.assertTrue(set(pos_pairs) == set(pos_pairs_new))

                    self.assertTrue(torch.equal(a2_new, a2))
                    self.assertTrue(torch.equal(n_new, n))

    def test_memory_filled_each_batch_matches_inner_loss(self):
        for dtype in TEST_DTYPES:
            for test_enqueue_mask in [False, True]:
                for memory_size in range(20, 40, 5):
                    inner_loss = NTXentLoss(temperature=0.1)
                    inner_miner = TripletMarginMiner(margin=0.1)
                    loss = CrossBatchMemory(
                        loss=inner_loss,
                        embedding_size=self.embedding_size,
                        memory_size=memory_size,
                    ).to(TEST_DEVICE, dtype)
                    loss_with_miner = CrossBatchMemory(
                        loss=inner_loss,
                        embedding_size=self.embedding_size,
                        memory_size=memory_size,
                        miner=inner_miner,
                    ).to(TEST_DEVICE, dtype)
                    for _ in range(10):
                        if test_enqueue_mask:
                            enqueue_mask = torch.zeros(
                                memory_size * 2, dtype=torch.bool, device=TEST_DEVICE
                            )
                            enqueue_mask[memory_size:] = True
                            batch_size = memory_size * 2
                        else:
                            enqueue_mask = None
                            batch_size = memory_size
                        embeddings = (
                            torch.randn(batch_size, self.embedding_size)
                            .to(TEST_DEVICE)
                            .type(dtype)
                        )
                        labels = torch.randint(0, 4, (batch_size,)).to(TEST_DEVICE)

                        if enqueue_mask is not None:
                            not_enqueue_emb = embeddings[~enqueue_mask]
                            not_enqueue_labels = labels[~enqueue_mask]
                            enqueue_emb = embeddings[enqueue_mask]
                            enqueue_labels = labels[enqueue_mask]
                            pairs = indexing.get_all_pairs_indices(
                                not_enqueue_labels, enqueue_labels
                            )
                            inner_loss_val = inner_loss(
                                not_enqueue_emb,
                                not_enqueue_labels,
                                pairs,
                                enqueue_emb,
                                enqueue_labels,
                            )
                        else:
                            inner_loss_val = inner_loss(embeddings, labels)
                        loss_val = loss(embeddings, labels, enqueue_mask=enqueue_mask)
                        self.assertTrue(torch.isclose(inner_loss_val, loss_val))

                        if test_enqueue_mask:
                            triplets = inner_miner(
                                not_enqueue_emb,
                                not_enqueue_labels,
                                ref_emb=enqueue_emb,
                                ref_labels=enqueue_labels,
                            )
                            inner_loss_val = inner_loss(
                                not_enqueue_emb,
                                not_enqueue_labels,
                                triplets,
                                enqueue_emb,
                                enqueue_labels,
                            )
                        else:
                            triplets = inner_miner(embeddings, labels)
                            inner_loss_val = inner_loss(embeddings, labels, triplets)
                        loss_val = loss_with_miner(
                            embeddings, labels, enqueue_mask=enqueue_mask
                        )
                        self.assertTrue(torch.isclose(inner_loss_val, loss_val))

    def test_with_distance_weighted_miner(self):
        for dtype in TEST_DTYPES:
            memory_size = 256
            inner_loss = NTXentLoss(temperature=0.1)
            inner_miner = DistanceWeightedMiner(cutoff=0.5, nonzero_loss_cutoff=1.4)
            loss_with_miner = CrossBatchMemory(
                loss=inner_loss,
                embedding_size=2,
                memory_size=memory_size,
                miner=inner_miner,
            ).to(TEST_DEVICE, dtype)
            for _ in range(20):
                embedding_angles = torch.arange(0, 32)
                embeddings = torch.tensor(
                    [angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(TEST_DEVICE)
                labels = torch.randint(low=0, high=10, size=(32,)).to(TEST_DEVICE)
                loss_val = loss_with_miner(embeddings, labels)
                loss_val.backward()
                self.assertTrue(True)

    def test_loss(self):
        for dtype in TEST_DTYPES:
            num_labels = 10
            num_iter = 10
            batch_size = 32
            for inner_loss in [ContrastiveLoss(), MultiSimilarityLoss()]:
                inner_miner = MultiSimilarityMiner(0.3)
                self.loss = CrossBatchMemory(
                    loss=inner_loss,
                    embedding_size=self.embedding_size,
                    memory_size=self.memory_size,
                ).to(TEST_DEVICE, dtype)
                self.loss_with_miner = CrossBatchMemory(
                    loss=inner_loss,
                    miner=inner_miner,
                    embedding_size=self.embedding_size,
                    memory_size=self.memory_size,
                ).to(TEST_DEVICE, dtype)
                all_embeddings = torch.tensor([], dtype=dtype).to(TEST_DEVICE)
                all_labels = torch.tensor([], dtype=torch.long).to(TEST_DEVICE)
                for _ in range(num_iter):
                    embeddings = (
                        torch.randn(batch_size, self.embedding_size)
                        .to(TEST_DEVICE)
                        .type(dtype)
                    )
                    labels = torch.randint(0, num_labels, (batch_size,)).to(TEST_DEVICE)
                    loss = self.loss(embeddings, labels)
                    loss_with_miner = self.loss_with_miner(embeddings, labels)
                    all_embeddings = torch.cat([all_embeddings, embeddings])
                    all_labels = torch.cat([all_labels, labels]).to(TEST_DEVICE)

                    indices_tuple = indexing.get_all_pairs_indices(labels, all_labels)
                    a1, p, a2, n = indexing.remove_self_comparisons(
                        indices_tuple, recent(all_labels, batch_size)
                    )
                    correct_loss = inner_loss(
                        embeddings,
                        labels,
                        (a1, p, a2, n),
                        all_embeddings,
                        all_labels,
                    )
                    self.assertTrue(torch.isclose(loss, correct_loss))

                    indices_tuple = inner_miner(
                        embeddings,
                        labels,
                        ref_emb=all_embeddings,
                        ref_labels=all_labels,
                    )
                    a1, p, a2, n = indexing.remove_self_comparisons(
                        indices_tuple, recent(all_labels, batch_size)
                    )
                    correct_loss_with_miner = inner_loss(
                        embeddings,
                        labels,
                        (a1, p, a2, n),
                        all_embeddings,
                        all_labels,
                    )
                    self.assertTrue(
                        torch.isclose(loss_with_miner, correct_loss_with_miner)
                    )

    def test_queue(self):
        for test_enqueue_mask in [False, True]:
            for dtype in TEST_DTYPES:
                batch_size = 32
                enqueue_batch_size = 15
                self.loss = CrossBatchMemory(
                    loss=ContrastiveLoss(),
                    embedding_size=self.embedding_size,
                    memory_size=self.memory_size,
                ).to(TEST_DEVICE, dtype)
                for i in range(30):
                    embeddings = (
                        torch.randn(batch_size, self.embedding_size)
                        .to(TEST_DEVICE)
                        .type(dtype)
                    )
                    labels = torch.arange(batch_size).to(TEST_DEVICE)
                    q = self.loss.queue_idx
                    B = enqueue_batch_size if test_enqueue_mask else batch_size
                    if test_enqueue_mask:
                        enqueue_mask = torch.zeros(
                            batch_size, dtype=torch.bool, device=TEST_DEVICE
                        )
                        enqueue_mask[:-2:2] = True
                    else:
                        enqueue_mask = None

                    self.assertTrue(q == (i * B) % self.memory_size)
                    self.loss(embeddings, labels, enqueue_mask=enqueue_mask)

                    start_idx = q
                    if q + B == self.memory_size:
                        end_idx = self.memory_size
                    else:
                        end_idx = (q + B) % self.memory_size
                    if start_idx < end_idx:
                        if test_enqueue_mask:
                            self.assertTrue(
                                torch.equal(
                                    embeddings[enqueue_mask],
                                    self.loss.embedding_memory[start_idx:end_idx],
                                )
                            )
                            self.assertTrue(
                                torch.equal(
                                    labels[enqueue_mask],
                                    self.loss.label_memory[start_idx:end_idx],
                                )
                            )
                        else:
                            self.assertTrue(
                                torch.equal(
                                    embeddings,
                                    self.loss.embedding_memory[start_idx:end_idx],
                                )
                            )
                            self.assertTrue(
                                torch.equal(
                                    labels, self.loss.label_memory[start_idx:end_idx]
                                )
                            )
                    else:
                        correct_embeddings = torch.cat(
                            [
                                self.loss.embedding_memory[start_idx:],
                                self.loss.embedding_memory[:end_idx],
                            ],
                            dim=0,
                        )
                        correct_labels = torch.cat(
                            [
                                self.loss.label_memory[start_idx:],
                                self.loss.label_memory[:end_idx],
                            ],
                            dim=0,
                        ).to(TEST_DEVICE)
                        if test_enqueue_mask:
                            self.assertTrue(
                                torch.equal(
                                    embeddings[enqueue_mask], correct_embeddings
                                )
                            )
                            self.assertTrue(
                                torch.equal(labels[enqueue_mask], correct_labels)
                            )
                        else:
                            self.assertTrue(torch.equal(embeddings, correct_embeddings))
                            self.assertTrue(torch.equal(labels, correct_labels))

    def test_ref_mining(self):
        for dtype in TEST_DTYPES:
            batch_size = 32
            pair_miner = PairMarginMiner(pos_margin=0, neg_margin=1)
            triplet_miner = TripletMarginMiner(margin=1)
            self.loss = CrossBatchMemory(
                loss=ContrastiveLoss(),
                embedding_size=self.embedding_size,
                memory_size=self.memory_size,
            ).to(TEST_DEVICE, dtype)
            for _ in range(30):
                embeddings = (
                    torch.randn(batch_size, self.embedding_size)
                    .to(TEST_DEVICE)
                    .type(dtype)
                )
                labels = torch.arange(batch_size).to(TEST_DEVICE)
                self.loss(embeddings, labels)

                a1, p, a2, n = indexing.get_all_pairs_indices(
                    labels, self.loss.label_memory
                )
                self.assertTrue(
                    not torch.any((labels[a1] - self.loss.label_memory[p]).bool())
                )
                self.assertTrue(
                    torch.all((labels[a2] - self.loss.label_memory[n]).bool())
                )

                a1, p, a2, n = pair_miner(
                    embeddings,
                    labels,
                    ref_emb=self.loss.embedding_memory,
                    ref_labels=self.loss.label_memory,
                )
                self.assertTrue(
                    not torch.any((labels[a1] - self.loss.label_memory[p]).bool())
                )
                self.assertTrue(
                    torch.all((labels[a2] - self.loss.label_memory[n]).bool())
                )

                a, p, n = triplet_miner(
                    embeddings,
                    labels,
                    ref_emb=self.loss.embedding_memory,
                    ref_labels=self.loss.label_memory,
                )
                self.assertTrue(
                    not torch.any((labels[a] - self.loss.label_memory[p]).bool())
                )
                self.assertTrue(
                    torch.all((labels[a] - self.loss.label_memory[n]).bool())
                )

    def test_input_indices_tuple(self):
        for dtype in TEST_DTYPES:
            batch_size = 32
            pair_miner = PairMarginMiner(pos_margin=0, neg_margin=1)
            triplet_miner = TripletMarginMiner(margin=1)
            self.loss = CrossBatchMemory(
                loss=ContrastiveLoss(),
                embedding_size=self.embedding_size,
                memory_size=self.memory_size,
            ).to(TEST_DEVICE, dtype)
            for _ in range(30):
                embeddings = (
                    torch.randn(batch_size, self.embedding_size)
                    .to(TEST_DEVICE)
                    .type(dtype)
                )
                labels = torch.arange(batch_size).to(TEST_DEVICE)
                self.loss(embeddings, labels)
                for curr_miner in [pair_miner, triplet_miner]:
                    input_indices_tuple = curr_miner(embeddings, labels)
                    a1ii, pii, a2ii, nii = indexing.convert_to_pairs(
                        input_indices_tuple, labels
                    )
                    indices_tuple = indexing.get_all_pairs_indices(
                        labels, self.loss.label_memory
                    )
                    slots = (
                        torch.arange(batch_size, device=TEST_DEVICE)
                        + self.loss.queue_idx
                        - batch_size
                    ) % self.memory_size
                    a1i, pi, a2i, ni = indexing.remove_self_comparisons(
                        indices_tuple, slots
                    )
                    a1, p, a2, n = self.loss.create_indices_tuple(
                        embeddings,
                        labels,
                        self.loss.embedding_memory,
                        self.loss.label_memory,
                        input_indices_tuple,
                        slots,
                    )
                    self.assertTrue(
                        not torch.any((labels[a1] - self.loss.label_memory[p]).bool())
                    )
                    self.assertTrue(
                        torch.all((labels[a2] - self.loss.label_memory[n]).bool())
                    )
                    self.assertTrue(torch.all(a1 == torch.cat([a1i, a1ii])))
                    self.assertTrue(torch.all(p == torch.cat([pi, pii])))
                    self.assertTrue(torch.all(a2 == torch.cat([a2i, a2ii])))
                    self.assertTrue(torch.all(n == torch.cat([ni, nii])))

    def test_all_losses(self):
        for dtype in TEST_DTYPES:
            num_labels = 10
            num_iter = 10
            batch_size = 32
            for inner_loss in self.load_valid_loss_fns():
                self.loss = CrossBatchMemory(
                    loss=inner_loss,
                    embedding_size=self.embedding_size,
                    memory_size=self.memory_size,
                ).to(TEST_DEVICE, dtype)

                all_embeddings = torch.tensor([], dtype=dtype).to(TEST_DEVICE)
                all_labels = torch.tensor([], dtype=torch.long).to(TEST_DEVICE)
                for _ in range(num_iter):
                    embeddings = torch.randn(
                        batch_size, self.embedding_size, device=TEST_DEVICE, dtype=dtype
                    )
                    labels = torch.randint(
                        0, num_labels, (batch_size,), device=TEST_DEVICE
                    )

                    loss = self.loss(embeddings, labels)

                    all_embeddings = torch.cat([all_embeddings, embeddings])
                    all_labels = torch.cat([all_labels, labels]).to(TEST_DEVICE)

                    indices_tuple = indexing.get_all_pairs_indices(labels, all_labels)
                    a1, p, a2, n = indexing.remove_self_comparisons(
                        indices_tuple, recent(all_labels, batch_size)
                    )

                    correct_loss = inner_loss(
                        embeddings,
                        labels,
                        (a1, p, a2, n),
                        all_embeddings,
                        all_labels,
                    )
                    self.assertTrue(torch.isclose(loss, correct_loss))

    def load_valid_loss_fns(self):
        loss_fns = [
            losses.AngularLoss(),
            losses.CircleLoss(),
            losses.ContrastiveLoss(),
            losses.GeneralizedLiftedStructureLoss(),
            losses.IntraPairVarianceLoss(),
            losses.MarginLoss(),
            losses.MultiSimilarityLoss(),
            losses.NCALoss(),
            losses.NTXentLoss(),
            losses.SignalToNoiseRatioContrastiveLoss(),
            losses.SupConLoss(),
            losses.TripletMarginLoss(),
            losses.TupletMarginLoss(),
        ]

        return loss_fns

    def test_reset_queue(self):
        self.loss = CrossBatchMemory(
            loss=ContrastiveLoss(),
            embedding_size=self.embedding_size,
            memory_size=self.memory_size,
        ).to(TEST_DEVICE)

        init_emb = torch.zeros(
            self.memory_size, self.embedding_size, device=TEST_DEVICE
        )
        init_label = torch.zeros(self.memory_size, dtype=torch.long, device=TEST_DEVICE)
        self.assertTrue(torch.equal(self.loss.embedding_memory, init_emb))
        self.assertTrue(torch.equal(self.loss.label_memory, init_label))

        self.loss(
            torch.randn(32, 128, device=TEST_DEVICE),
            torch.randint(0, 2, size=(32,), device=TEST_DEVICE),
        )
        self.assertTrue(not torch.equal(self.loss.embedding_memory, init_emb))
        self.assertTrue(not torch.equal(self.loss.label_memory, init_label))

        self.loss.reset_queue()
        self.assertTrue(torch.equal(self.loss.embedding_memory, init_emb))
        self.assertTrue(torch.equal(self.loss.label_memory, init_label))

    def test_reset_queue_keeps_placement(self):
        loss = CrossBatchMemory(
            loss=ContrastiveLoss(),
            embedding_size=2,
            memory_size=4,
            label_dtype=torch.int32,
        ).to(TEST_DEVICE, torch.float64)
        loss(
            torch.randn(3, 2, device=TEST_DEVICE, dtype=torch.float64),
            torch.tensor([0, 0, 1], device=TEST_DEVICE),
        )
        loss.reset_queue()
        self.assertEqual(loss.embedding_memory.dtype, torch.float64)
        self.assertEqual(loss.embedding_memory.device.type, TEST_DEVICE.type)
        self.assertEqual(loss.label_memory.dtype, torch.int32)
        self.assertFalse(loss.embedding_memory.any())
        self.assertFalse(loss.label_memory.any())
        self.assertEqual((loss.queue_idx, loss.has_been_filled), (0, False))


class TestCrossBatchMemorySelectingMiner(unittest.TestCase):
    def test_easy_positive_miner_skips_each_anchors_queued_copy(self):
        distance = LpDistance(normalize_embeddings=False)
        miner = BatchEasyHardMiner(
            pos_strategy="easy", neg_strategy="hard", distance=distance
        )
        memory = CrossBatchMemory(
            NTXentLoss(), embedding_size=2, memory_size=8, miner=miner
        ).to(TEST_DEVICE, torch.float64)
        batches = [
            [[1.0, 0.0], [0.9, 0.3], [-1.0, 0.0], [-0.8, 0.4]],
            [[0.7, -0.6], [0.2, 1.0], [-0.3, -1.0], [-0.6, 0.7]],
        ]
        labels = torch.tensor([0, 0, 1, 1], device=TEST_DEVICE)
        stored = torch.empty(0, 2, device=TEST_DEVICE, dtype=torch.float64)
        stored_labels = labels[:0]
        for batch in batches:
            embeddings = torch.tensor(
                batch, device=TEST_DEVICE, dtype=torch.float64, requires_grad=True
            )
            own = torch.arange(len(stored), len(stored) + 4, device=TEST_DEVICE)
            stored = torch.cat((stored, embeddings.detach()))
            stored_labels = torch.cat((stored_labels, labels))
            distances = torch.cdist(embeddings.detach(), stored)
            same = labels[:, None] == stored_labels[None]
            positive = same.clone()
            positive[torch.arange(4, device=TEST_DEVICE), own] = False
            anchors = torch.arange(4, device=TEST_DEVICE)
            pairs = PairIndices(
                anchors,
                distances.masked_fill(~positive, torch.inf).argmin(dim=1),
                anchors,
                distances.masked_fill(same, torch.inf).argmin(dim=1),
            )
            expected = NTXentLoss()(embeddings, labels, pairs, stored, stored_labels)
            expected_grad = torch.autograd.grad(expected, embeddings)[0]
            loss = memory(embeddings, labels)
            loss.backward()
            self.assertGreater(loss.item(), 0)
            torch.testing.assert_close(loss, expected)
            torch.testing.assert_close(embeddings.grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
