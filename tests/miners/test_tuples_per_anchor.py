import unittest

import torch

from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.types import PairIndices, TripletIndices

from .. import TEST_DEVICE

MINERS = {
    "AngularMiner": lambda **kw: miners.AngularMiner(alpha=1, **kw),
    "BatchEasyHardMiner": lambda **kw: miners.BatchEasyHardMiner(
        pos_strategy="all", neg_strategy="all", **kw
    ),
    "BatchHardMiner": lambda **kw: miners.BatchHardMiner(**kw),
    "DistanceWeightedMiner": lambda **kw: miners.DistanceWeightedMiner(**kw),
    "HDCMiner": lambda **kw: miners.HDCMiner(**kw),
    "MultiSimilarityMiner": lambda **kw: miners.MultiSimilarityMiner(epsilon=2, **kw),
    "PairMarginMiner": lambda **kw: miners.PairMarginMiner(
        pos_margin=0, neg_margin=4, **kw
    ),
    "TripletMarginMiner": lambda **kw: miners.TripletMarginMiner(margin=4, **kw),
    "UniformHistogramMiner": lambda **kw: miners.UniformHistogramMiner(**kw),
}


def batch():
    generator = torch.Generator().manual_seed(6)
    embeddings = torch.randn(24, 6, generator=generator).to(TEST_DEVICE)
    return embeddings, (torch.arange(24) // 4).to(TEST_DEVICE)


def rows(tensors):
    return torch.stack(tensors, dim=1).tolist()


def most_per_anchor(anchors):
    return int(torch.bincount(anchors.cpu(), minlength=1).max())


class TestTuplesPerAnchor(unittest.TestCase):
    def test_every_exported_miner_applies_the_bound(self):
        exported = {
            name
            for name, value in vars(miners).items()
            if isinstance(value, type)
            and issubclass(value, miners.BaseMiner)
            and value is not miners.BaseMiner
        }
        self.assertEqual(exported, MINERS.keys())
        embeddings, labels = batch()
        for name, factory in MINERS.items():
            for limit in (1, 2, 5):
                with self.subTest(miner=name, limit=limit):
                    torch.manual_seed(0)
                    full = factory()(embeddings, labels)
                    torch.manual_seed(0)
                    bounded = factory(tuples_per_anchor=limit)(embeddings, labels)
                    self.assertIs(type(bounded), type(full))
                    if isinstance(bounded, TripletIndices):
                        groups = [(full, bounded)]
                    else:
                        groups = [
                            (full[:2], bounded[:2]),
                            (full[2:], bounded[2:]),
                        ]
                    for mined, kept in groups:
                        anchors = kept[0]
                        if len(anchors):
                            self.assertLessEqual(most_per_anchor(anchors), limit)
                        full_rows = rows(mined)
                        expected = sum(
                            min(limit, int(count))
                            for count in torch.unique(mined[0], return_counts=True)[1]
                        )
                        self.assertEqual(len(anchors), expected)
                        self.assertTrue(all(x in full_rows for x in rows(kept)))

    def test_limit_above_every_count_keeps_all_tuples_in_order(self):
        embeddings, labels = batch()
        torch.manual_seed(0)
        full = miners.TripletMarginMiner(margin=4)(embeddings, labels)
        bounded = miners.TripletMarginMiner(margin=4, tuples_per_anchor=10**6)(
            embeddings, labels
        )
        for x, y in zip(bounded, full, strict=True):
            self.assertTrue(torch.equal(x, y))

    def test_kept_tuples_stay_in_mined_order(self):
        embeddings, labels = batch()
        full = miners.PairMarginMiner(pos_margin=0, neg_margin=4)(embeddings, labels)
        bounded = miners.PairMarginMiner(
            pos_margin=0, neg_margin=4, tuples_per_anchor=2
        )(embeddings, labels)
        positions = [rows(full[:2]).index(row) for row in rows(bounded[:2])]
        self.assertEqual(positions, sorted(positions))

    def test_each_tuple_is_equally_likely(self):
        embeddings = torch.zeros(4, 2, device=TEST_DEVICE)
        labels = torch.tensor([0, 0, 0, 1], device=TEST_DEVICE)
        miner = miners.BatchEasyHardMiner(
            pos_strategy="all", neg_strategy="all", tuples_per_anchor=1
        )
        counts = torch.zeros(4, dtype=torch.long)
        trials = 3000
        for _ in range(trials):
            a1, p, _, _ = miner(embeddings, labels)
            chosen = p[a1 == 0].cpu()
            self.assertEqual(len(chosen), 1)
            counts[chosen] += 1
        self.assertEqual(counts[[0, 3]].tolist(), [0, 0])
        self.assertLess(abs(counts[1].item() - trials / 2), 4 * (trials / 4) ** 0.5)

    def test_no_tuple_repeats(self):
        embeddings, labels = batch()
        miner = miners.TripletMarginMiner(margin=4, tuples_per_anchor=7)
        triplets = rows(miner(embeddings, labels))
        self.assertEqual(len(triplets), len({tuple(x) for x in triplets}))

    def test_seed_repeats_selection(self):
        embeddings, labels = batch()
        miner = miners.TripletMarginMiner(margin=4, tuples_per_anchor=2)
        torch.manual_seed(5)
        first = miner(embeddings, labels)
        torch.manual_seed(5)
        second = miner(embeddings, labels)
        for x, y in zip(first, second, strict=True):
            self.assertTrue(torch.equal(x, y))

    def test_empty_output_stays_empty(self):
        embeddings, labels = batch()
        for miner, kind in (
            (
                miners.TripletMarginMiner(
                    margin=0, type_of_triplets="easy", tuples_per_anchor=2
                ),
                TripletIndices,
            ),
            (
                miners.PairMarginMiner(
                    pos_margin=9, neg_margin=-9, tuples_per_anchor=2
                ),
                PairIndices,
            ),
        ):
            with self.subTest(kind=kind.__name__):
                output = miner(embeddings, labels[: len(embeddings)])
                self.assertIsInstance(output, kind)
        empty = embeddings[:0]
        output = miners.PairMarginMiner(tuples_per_anchor=2)(empty, labels[:0])
        self.assertEqual([len(x) for x in output], [0, 0, 0, 0])

    def test_reference_set_and_precomputed_matrix(self):
        embeddings, labels = batch()
        reference = embeddings[:10] + 0.1
        ref_labels = labels[:10]
        miner = miners.PairMarginMiner(pos_margin=0, neg_margin=4, tuples_per_anchor=2)
        a1, p, a2, n = miner(
            embeddings,
            labels,
            ref_emb=reference,
            ref_labels=ref_labels,
            mat=miner.distance(embeddings, reference),
        )
        self.assertLessEqual(most_per_anchor(a1), 2)
        self.assertLessEqual(most_per_anchor(a2), 2)
        self.assertTrue(torch.equal(labels[a1], ref_labels[p]))
        self.assertTrue((labels[a2] != ref_labels[n]).all())

    def test_loss_reads_bounded_tuples(self):
        embeddings, labels = batch()
        embeddings = embeddings.clone().requires_grad_()
        miner = miners.TripletMarginMiner(margin=4, tuples_per_anchor=3)
        loss = losses.TripletMarginLoss(margin=1)(
            embeddings, labels, miner(embeddings, labels)
        )
        loss.backward()
        assert embeddings.grad is not None
        self.assertTrue(embeddings.grad.any())

    def test_invalid_limits(self):
        for limit in (0, -1):
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(ValueError, "tuples_per_anchor"),
            ):
                miners.HDCMiner(tuples_per_anchor=limit)
