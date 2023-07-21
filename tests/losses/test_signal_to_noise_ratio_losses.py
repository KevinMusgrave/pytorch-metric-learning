import unittest

import torch

from pytorch_metric_learning.losses import SignalToNoiseRatioContrastiveLoss
from pytorch_metric_learning.regularizers import ZeroMeanRegularizer

from .. import TEST_DEVICE, TEST_DTYPES
from ..zzz_testing_utils.testing_utils import angle_to_coord


class TestSNRContrastiveLoss(unittest.TestCase):
    def test_snr_contrastive_loss(self):
        pos_margin, neg_margin, reg_weight = 0, 0.1, 0.1
        loss_func = SignalToNoiseRatioContrastiveLoss(
            pos_margin=pos_margin,
            neg_margin=neg_margin,
            regularizer=ZeroMeanRegularizer(),
            reg_weight=reg_weight,
        )

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

            loss = loss_func(embeddings, labels)
            loss.backward()

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

            normalized_embeddings = torch.nn.functional.normalize(embeddings)
            correct_pos_loss = 0
            correct_neg_loss = 0
            num_non_zero = 0
            for a, p in pos_pairs:
                anchor, positive = normalized_embeddings[a], normalized_embeddings[p]
                curr_loss = torch.relu(
                    torch.var(anchor - positive) / torch.var(anchor) - pos_margin
                )
                correct_pos_loss += curr_loss
                if curr_loss > 0:
                    num_non_zero += 1
            if num_non_zero > 0:
                correct_pos_loss /= num_non_zero

            num_non_zero = 0
            for a, n in neg_pairs:
                anchor, negative = normalized_embeddings[a], normalized_embeddings[n]
                curr_loss = torch.relu(
                    neg_margin - torch.var(anchor - negative) / torch.var(anchor)
                )
                correct_neg_loss += curr_loss
                if curr_loss > 0:
                    num_non_zero += 1
            if num_non_zero > 0:
                correct_neg_loss /= num_non_zero

            reg_loss = torch.mean(torch.abs(torch.sum(embeddings, dim=1)))

            correct_total = correct_pos_loss + correct_neg_loss + reg_weight * reg_loss
            rtol = 1e-2 if dtype == torch.float16 else 1e-5
            self.assertTrue(torch.isclose(loss, correct_total, rtol=rtol))

    def test_with_no_valid_pairs(self):
        reg_weight = 0.1
        loss_func = SignalToNoiseRatioContrastiveLoss(
            pos_margin=0,
            neg_margin=0.5,
            regularizer=ZeroMeanRegularizer(),
            reg_weight=reg_weight,
        )
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
            reg_loss = torch.mean(torch.abs(torch.sum(embeddings, dim=1))) * reg_weight
            loss = loss_func(embeddings, labels)
            loss.backward()
            self.assertEqual(loss, reg_loss)
