import unittest

import torch

from pytorch_metric_learning.losses import MultiSimilarityLoss

from .. import TEST_DEVICE, TEST_DTYPES, helpers
from ..helpers import angle_to_coord
from .utils import get_pair_embeddings_with_ref


class TestMultiSimilarityLoss(unittest.TestCase):
    def test_multi_similarity_loss(self):
        for dtype in TEST_DTYPES:
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long).to(TEST_DEVICE)
            pos_pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]
            neg_pairs = [
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 0),
                (2, 1),
                (2, 4),
                (3, 0),
                (3, 1),
                (3, 4),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 3),
            ]
            self.helper(embeddings, labels, pos_pairs, neg_pairs, dtype)

    def test_multi_similarity_loss_with_ref(self):
        for dtype in TEST_DTYPES:
            (
                embeddings,
                labels,
                ref_emb,
                ref_labels,
                pos_pairs,
                neg_pairs,
            ) = get_pair_embeddings_with_ref(dtype, TEST_DEVICE)
            self.helper(
                embeddings, labels, pos_pairs, neg_pairs, dtype, ref_emb, ref_labels
            )

    def helper(
        self,
        embeddings,
        labels,
        pos_pairs,
        neg_pairs,
        dtype,
        ref_emb=None,
        ref_labels=None,
    ):
        if dtype == torch.float16:
            alpha, beta, base = 0.1, 10, 0.5
        else:
            alpha, beta, base = 0.1, 40, 0.5
        loss_func = MultiSimilarityLoss(alpha=alpha, beta=beta, base=base)
        loss = loss_func(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
        loss.backward()
        embeddings = torch.nn.functional.normalize(embeddings)
        correct_total = torch.zeros((), dtype=dtype, device=TEST_DEVICE)
        for i in range(len(embeddings)):
            correct_pos_loss = torch.zeros((), dtype=dtype, device=TEST_DEVICE)
            correct_neg_loss = torch.zeros((), dtype=dtype, device=TEST_DEVICE)
            for a, p in pos_pairs:
                if a == i:
                    anchor = embeddings[a]
                    positive = embeddings[p] if ref_emb is None else ref_emb[p]
                    correct_pos_loss += torch.exp(
                        -alpha * (torch.matmul(anchor, positive) - base)
                    )
            if correct_pos_loss > 0:
                correct_pos_loss = (1 / alpha) * torch.log(1 + correct_pos_loss)

            for a, n in neg_pairs:
                if a == i:
                    anchor = embeddings[a]
                    negative = embeddings[n] if ref_emb is None else ref_emb[n]
                    correct_neg_loss += torch.exp(
                        beta * (torch.matmul(anchor, negative) - base)
                    )
            if correct_neg_loss > 0:
                correct_neg_loss = (1 / beta) * torch.log(1 + correct_neg_loss)
            correct_total += correct_pos_loss + correct_neg_loss

        correct_total /= embeddings.size(0)
        rtol = helpers.rtol(dtype)
        self.assertTrue(torch.isclose(loss, correct_total, rtol=rtol))

    def test_with_no_valid_pairs(self):
        alpha, beta, base = 0.1, 40, 0.5
        loss_func = MultiSimilarityLoss(alpha=alpha, beta=beta, base=base)
        for dtype in TEST_DTYPES:
            embedding_angles = [0]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.tensor([0], dtype=torch.long).to(TEST_DEVICE)
            loss = loss_func(embeddings, labels)
            loss.backward()
            self.assertEqual(loss, 0)

    def test_anchors_without_mined_pairs_are_excluded(self):
        alpha, beta, base = 2, 10, 0.5
        labels = torch.tensor([0, 0, 1, 1, 2, 0], device=TEST_DEVICE)
        pos_pairs = [(0, 1), (0, 5), (2, 3)]
        neg_pairs = [(0, 2), (3, 4), (4, 0)]
        mined = tuple(
            torch.tensor(column, device=TEST_DEVICE)
            for pairs in (pos_pairs, neg_pairs)
            for column in zip(*pairs, strict=True)
        )
        torch.manual_seed(8)
        embeddings = torch.randn(6, 3)

        def reference(x):
            x = torch.nn.functional.normalize(x, dim=1)
            sims = x @ x.T
            losses = []
            for a in range(len(labels)):
                pos = [sims[a, p] for anchor, p in pos_pairs if anchor == a]
                neg = [sims[a, n] for anchor, n in neg_pairs if anchor == a]
                if not pos and not neg:
                    continue
                loss = sims.new_zeros(())
                if pos:
                    pos_sum = torch.exp(-alpha * (torch.stack(pos) - base)).sum()
                    loss = loss + torch.log(1 + pos_sum) / alpha
                if neg:
                    neg_sum = torch.exp(beta * (torch.stack(neg) - base)).sum()
                    loss = loss + torch.log(1 + neg_sum) / beta
                losses.append(loss)
            return torch.stack(losses).mean()

        loss_func = MultiSimilarityLoss(alpha=alpha, beta=beta, base=base)
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                helpers.assert_matches_float64(
                    lambda x: loss_func(x, labels, mined),
                    reference,
                    [embeddings],
                    dtype,
                )
