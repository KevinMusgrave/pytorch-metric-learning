import unittest

import torch

from pytorch_metric_learning.losses import MultiSimilarityLoss
from ..zzz_testing_utils.testing_utils import angle_to_coord

from .. import TEST_DEVICE, TEST_DTYPES
from .utils import get_pair_embeddings_with_ref


class TestMultiSimilarityLoss(unittest.TestCase):
    def test_multi_similarity_loss(self):
        for dtype in TEST_DTYPES:

            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.LongTensor([0, 0, 1, 1, 2])
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
        correct_total = 0
        for i in range(len(embeddings)):
            correct_pos_loss = 0
            correct_neg_loss = 0
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
        rtol = 1e-2 if dtype == torch.float16 else 1e-5
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
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.LongTensor([0])
            loss = loss_func(embeddings, labels)
            loss.backward()
            self.assertEqual(loss, 0)
