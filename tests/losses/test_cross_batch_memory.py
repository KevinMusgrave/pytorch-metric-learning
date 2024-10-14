import unittest

import torch

import pytorch_metric_learning.losses as losses
from pytorch_metric_learning.losses import (
    ContrastiveLoss,
    CrossBatchMemory,
    MultiSimilarityLoss,
    NTXentLoss,
)
from pytorch_metric_learning.miners import (
    DistanceWeightedMiner,
    MultiSimilarityMiner,
    PairMarginMiner,
    TripletMarginMiner,
)
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from .. import TEST_DEVICE, TEST_DTYPES
from ..zzz_testing_utils.testing_utils import angle_to_coord


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
            )
            loss.embedding_memory = loss.embedding_memory.to(TEST_DEVICE).type(dtype)
            loss.label_memory = loss.label_memory.to(TEST_DEVICE)
            embeddings = (
                torch.randn(batch_size, self.embedding_size).to(TEST_DEVICE).type(dtype)
            )
            labels = torch.randint(0, 10, (batch_size,)).to(TEST_DEVICE)
            num_tuples = 1000
            num_non_identical = 147

            for i in range(30):
                loss.add_to_memory(embeddings, labels, batch_size)
                for identical in [True, False]:
                    # triplets
                    a = torch.randint(0, batch_size, (num_tuples,)).to(TEST_DEVICE)
                    p = a + loss.curr_batch_idx[0]
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
                    a_new, p_new, n_new = lmu.remove_self_comparisons(
                        (a, p, n), loss.curr_batch_idx, loss.memory_size
                    )
                    if identical:
                        self.assertTrue(len(a_new) == len(p_new) == len(n_new) == 0)
                    else:
                        triplets = set(
                            [
                                (x.item(), y.item(), z.item())
                                for x, y, z in zip(
                                    a[rand_diff_idx], p[rand_diff_idx], n[rand_diff_idx]
                                )
                            ]
                        )
                        triplets_new = set(
                            [
                                (x.item(), y.item(), z.item())
                                for x, y, z in zip(a_new, p_new, n_new)
                            ]
                        )
                        self.assertTrue(set(triplets) == set(triplets_new))

                    # pairs
                    a1 = torch.randint(0, batch_size, (num_tuples,)).to(TEST_DEVICE)
                    p = a1 + loss.curr_batch_idx[0]
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
                        + loss.curr_batch_idx[0]
                    ) % self.memory_size
                    a1_new, p_new, a2_new, n_new = lmu.remove_self_comparisons(
                        (a1, p, a2, n), loss.curr_batch_idx, loss.memory_size
                    )
                    if identical:
                        self.assertTrue(len(a1_new) == len(p_new) == 0)
                    else:
                        pos_pairs = set(
                            [
                                (x.item(), y.item())
                                for x, y in zip(a1[rand_diff_idx], p[rand_diff_idx])
                            ]
                        )
                        pos_pairs_new = set(
                            [(x.item(), y.item()) for x, y in zip(a1_new, p_new)]
                        )
                        self.assertTrue(set(pos_pairs) == set(pos_pairs_new))

                    self.assertTrue(torch.equal(a2_new, a2))
                    self.assertTrue(torch.equal(n_new, n))

    def test_sanity_check(self):
        # cross batch memory with batch_size == memory_size should be equivalent to just using the inner loss function
        for dtype in TEST_DTYPES:
            for test_enqueue_mask in [False, True]:
                for memory_size in range(20, 40, 5):
                    inner_loss = NTXentLoss(temperature=0.1)
                    inner_miner = TripletMarginMiner(margin=0.1)
                    loss = CrossBatchMemory(
                        loss=inner_loss,
                        embedding_size=self.embedding_size,
                        memory_size=memory_size,
                    )
                    loss_with_miner = CrossBatchMemory(
                        loss=inner_loss,
                        embedding_size=self.embedding_size,
                        memory_size=memory_size,
                        miner=inner_miner,
                    )
                    for i in range(10):
                        if test_enqueue_mask:
                            enqueue_mask = torch.zeros(memory_size * 2).bool()
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

                        if test_enqueue_mask:
                            not_enqueue_emb = embeddings[~enqueue_mask]
                            not_enqueue_labels = labels[~enqueue_mask]
                            enqueue_emb = embeddings[enqueue_mask]
                            enqueue_labels = labels[enqueue_mask]

                        if test_enqueue_mask:
                            pairs = lmu.get_all_pairs_indices(
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
                                enqueue_emb,
                                enqueue_labels,
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
            )
            for i in range(20):
                embedding_angles = torch.arange(0, 32)
                embeddings = torch.tensor(
                    [angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(
                    TEST_DEVICE
                )  # 2D embeddings
                labels = torch.randint(low=0, high=10, size=(32,)).to(TEST_DEVICE)
                loss_val = loss_with_miner(embeddings, labels)
                loss_val.backward()
                self.assertTrue(True)  # just check if we got here without an exception

    def test_loss(self):
        for dtype in TEST_DTYPES:
            num_labels = 10
            num_iter = 10
            batch_size = 32
            for inner_loss in [ContrastiveLoss(), MultiSimilarityLoss()]:
                inner_miner = MultiSimilarityMiner(0.3)
                outer_miner = MultiSimilarityMiner(0.2)
                self.loss = CrossBatchMemory(
                    loss=inner_loss,
                    embedding_size=self.embedding_size,
                    memory_size=self.memory_size,
                )
                self.loss_with_miner = CrossBatchMemory(
                    loss=inner_loss,
                    miner=inner_miner,
                    embedding_size=self.embedding_size,
                    memory_size=self.memory_size,
                )
                self.loss_with_miner2 = CrossBatchMemory(
                    loss=inner_loss,
                    miner=inner_miner,
                    embedding_size=self.embedding_size,
                    memory_size=self.memory_size,
                )
                all_embeddings = torch.tensor([], dtype=dtype).to(TEST_DEVICE)
                all_labels = torch.LongTensor([]).to(TEST_DEVICE)
                for i in range(num_iter):
                    embeddings = (
                        torch.randn(batch_size, self.embedding_size)
                        .to(TEST_DEVICE)
                        .type(dtype)
                    )
                    labels = torch.randint(0, num_labels, (batch_size,)).to(TEST_DEVICE)
                    loss = self.loss(embeddings, labels)
                    loss_with_miner = self.loss_with_miner(embeddings, labels)
                    oa1, op, oa2, on = outer_miner(embeddings, labels)
                    loss_with_miner_and_input_indices = self.loss_with_miner2(
                        embeddings, labels, (oa1, op, oa2, on)
                    )
                    all_embeddings = torch.cat([all_embeddings, embeddings])
                    all_labels = torch.cat([all_labels, labels])

                    # loss with no inner miner
                    indices_tuple = lmu.get_all_pairs_indices(labels, all_labels)
                    a1, p, a2, n = lmu.remove_self_comparisons(
                        indices_tuple, self.loss.curr_batch_idx, self.loss.memory_size
                    )
                    correct_loss = inner_loss(
                        embeddings,
                        labels,
                        (a1, p, a2, n),
                        all_embeddings,
                        all_labels,
                    )
                    self.assertTrue(torch.isclose(loss, correct_loss))

                    # loss with inner miner
                    indices_tuple = inner_miner(
                        embeddings, labels, all_embeddings, all_labels
                    )
                    a1, p, a2, n = lmu.remove_self_comparisons(
                        indices_tuple,
                        self.loss_with_miner.curr_batch_idx,
                        self.loss_with_miner.memory_size,
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

                    # loss with inner and outer miner
                    indices_tuple = inner_miner(
                        embeddings, labels, all_embeddings, all_labels
                    )
                    a1, p, a2, n = lmu.remove_self_comparisons(
                        indices_tuple,
                        self.loss_with_miner2.curr_batch_idx,
                        self.loss_with_miner2.memory_size,
                    )
                    a1 = torch.cat([oa1, a1])
                    p = torch.cat([op, p])
                    a2 = torch.cat([oa2, a2])
                    n = torch.cat([on, n])
                    correct_loss_with_miner_and_input_indice = inner_loss(
                        embeddings,
                        labels,
                        (a1, p, a2, n),
                        all_embeddings,
                        all_labels,
                    )
                    self.assertTrue(
                        torch.isclose(
                            loss_with_miner_and_input_indices,
                            correct_loss_with_miner_and_input_indice,
                        )
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
                )
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
                        enqueue_mask = torch.zeros(batch_size).bool()
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
                        )
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
            )
            for i in range(30):
                embeddings = (
                    torch.randn(batch_size, self.embedding_size)
                    .to(TEST_DEVICE)
                    .type(dtype)
                )
                labels = torch.arange(batch_size).to(TEST_DEVICE)
                self.loss(embeddings, labels)

                a1, p, a2, n = lmu.get_all_pairs_indices(labels, self.loss.label_memory)
                self.assertTrue(
                    not torch.any((labels[a1] - self.loss.label_memory[p]).bool())
                )
                self.assertTrue(
                    torch.all((labels[a2] - self.loss.label_memory[n]).bool())
                )

                a1, p, a2, n = pair_miner(
                    embeddings,
                    labels,
                    self.loss.embedding_memory,
                    self.loss.label_memory,
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
                    self.loss.embedding_memory,
                    self.loss.label_memory,
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
            )
            for i in range(30):
                embeddings = (
                    torch.randn(batch_size, self.embedding_size)
                    .to(TEST_DEVICE)
                    .type(dtype)
                )
                labels = torch.arange(batch_size).to(TEST_DEVICE)
                self.loss(embeddings, labels)
                for curr_miner in [pair_miner, triplet_miner]:
                    input_indices_tuple = curr_miner(embeddings, labels)
                    a1ii, pii, a2ii, nii = lmu.convert_to_pairs(
                        input_indices_tuple, labels
                    )
                    indices_tuple = lmu.get_all_pairs_indices(
                        labels, self.loss.label_memory
                    )
                    a1i, pi, a2i, ni = lmu.remove_self_comparisons(
                        indices_tuple, self.loss.curr_batch_idx, self.loss.memory_size
                    )
                    a1, p, a2, n = self.loss.create_indices_tuple(
                        embeddings,
                        labels,
                        self.loss.embedding_memory,
                        self.loss.label_memory,
                        input_indices_tuple,
                        True,
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
                )

                all_embeddings = torch.tensor([], dtype=dtype).to(TEST_DEVICE)
                all_labels = torch.LongTensor([]).to(TEST_DEVICE)
                for i in range(num_iter):
                    embeddings = torch.randn(
                        batch_size, self.embedding_size, device=TEST_DEVICE, dtype=dtype
                    )
                    labels = torch.randint(
                        0, num_labels, (batch_size,), device=TEST_DEVICE
                    )

                    loss = self.loss(embeddings, labels)

                    all_embeddings = torch.cat([all_embeddings, embeddings])
                    all_labels = torch.cat([all_labels, labels])

                    # loss with no inner miner
                    indices_tuple = lmu.get_all_pairs_indices(labels, all_labels)
                    a1, p, a2, n = lmu.remove_self_comparisons(
                        indices_tuple, self.loss.curr_batch_idx, self.loss.memory_size
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
        supported_losses = CrossBatchMemory.supported_losses()

        loss_fns = [
            losses.AngularLoss(),
            losses.CircleLoss(),
            losses.ContrastiveLoss(),
            losses.GeneralizedLiftedStructureLoss(),
            losses.IntraPairVarianceLoss(),
            losses.LiftedStructureLoss(),
            losses.MarginLoss(),
            losses.MultiSimilarityLoss(),
            losses.NCALoss(),
            losses.NTXentLoss(),
            losses.SignalToNoiseRatioContrastiveLoss(),
            losses.SupConLoss(),
            losses.TripletMarginLoss(),
            losses.TupletMarginLoss(),
        ]

        loaded_loss_names = [type(loss).__name__ for loss in loss_fns]
        assert set(loaded_loss_names) == set(supported_losses)

        return loss_fns

    def test_reset_queue(self):
        self.loss = CrossBatchMemory(
            loss=ContrastiveLoss(),
            embedding_size=self.embedding_size,
            memory_size=self.memory_size,
        )

        init_emb = torch.zeros(self.memory_size, self.embedding_size)
        init_label = torch.zeros(self.memory_size).long()
        self.assertTrue(torch.equal(self.loss.embedding_memory, init_emb))
        self.assertTrue(torch.equal(self.loss.label_memory, init_label))

        self.loss(torch.randn(32, 128), torch.randint(0, 2, size=(32,)))
        self.assertTrue(not torch.equal(self.loss.embedding_memory, init_emb))
        self.assertTrue(not torch.equal(self.loss.label_memory, init_label))

        self.loss.reset_queue()
        self.assertTrue(torch.equal(self.loss.embedding_memory, init_emb))
        self.assertTrue(torch.equal(self.loss.label_memory, init_label))


if __name__ == "__main__":
    unittest.main()
