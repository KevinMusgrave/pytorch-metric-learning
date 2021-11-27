import unittest

import torch

from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.reducers import MeanReducer

from .. import TEST_DEVICE, TEST_DTYPES
from ..zzz_testing_utils.testing_utils import angle_to_coord
from .utils import get_triplet_embeddings_with_ref


class TestTripletMarginLoss(unittest.TestCase):
    def test_triplet_margin_loss(self):
        for dtype in TEST_DTYPES:
            embeddings = torch.randn(5, 32, requires_grad=True, dtype=dtype,).to(
                TEST_DEVICE
            )  # 2D embeddings
            embeddings = torch.nn.functional.normalize(embeddings)
            labels = torch.LongTensor([0, 0, 1, 1, 2])

            triplets = [
                (0, 1, 2),
                (0, 1, 3),
                (0, 1, 4),
                (1, 0, 2),
                (1, 0, 3),
                (1, 0, 4),
                (2, 3, 0),
                (2, 3, 1),
                (2, 3, 4),
                (3, 2, 0),
                (3, 2, 1),
                (3, 2, 4),
            ]

            self.helper(embeddings, labels, triplets, dtype)

    def test_triplet_margin_loss_with_ref(self):
        for dtype in TEST_DTYPES:
            (
                embeddings,
                labels,
                ref_emb,
                ref_labels,
                triplets,
            ) = get_triplet_embeddings_with_ref(dtype, TEST_DEVICE)
            self.helper(embeddings, labels, triplets, dtype, ref_emb, ref_labels)

    def helper(
        self, embeddings, labels, triplets, dtype, ref_emb=None, ref_labels=None
    ):
        margin = 0.2
        loss_funcA = TripletMarginLoss(margin=margin)
        loss_funcB = TripletMarginLoss(margin=margin, reducer=MeanReducer())
        loss_funcC = TripletMarginLoss(margin=margin, distance=CosineSimilarity())
        loss_funcD = TripletMarginLoss(
            margin=margin, reducer=MeanReducer(), distance=CosineSimilarity()
        )
        loss_funcE = TripletMarginLoss(margin=margin, smooth_loss=True)

        [lossA, lossB, lossC, lossD, lossE] = [
            x(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
            for x in [loss_funcA, loss_funcB, loss_funcC, loss_funcD, loss_funcE]
        ]

        correct_loss = 0
        correct_loss_cosine = 0
        correct_smooth_loss = 0
        num_non_zero_triplets = 0
        num_non_zero_triplets_cosine = 0
        for a, p, n in triplets:
            anchor = embeddings[a]
            if ref_emb is not None:
                positive, negative = ref_emb[p], ref_emb[n]
            else:
                positive, negative = embeddings[p], embeddings[n]
            ap_dist = torch.sqrt(torch.sum((anchor - positive) ** 2))
            an_dist = torch.sqrt(torch.sum((anchor - negative) ** 2))
            curr_loss = torch.relu(ap_dist - an_dist + margin)
            curr_loss_cosine = torch.relu(
                torch.sum(anchor * negative) - torch.sum(anchor * positive) + margin
            )
            correct_smooth_loss += torch.nn.functional.softplus(
                ap_dist - an_dist + margin
            )
            if curr_loss > 0:
                num_non_zero_triplets += 1
            if curr_loss_cosine > 0:
                num_non_zero_triplets_cosine += 1
            correct_loss += curr_loss
            correct_loss_cosine += curr_loss_cosine
        rtol = 1e-2 if dtype == torch.float16 else 1e-5
        self.assertTrue(
            torch.isclose(lossA, correct_loss / num_non_zero_triplets, rtol=rtol)
        )
        self.assertTrue(torch.isclose(lossB, correct_loss / len(triplets), rtol=rtol))
        self.assertTrue(
            torch.isclose(
                lossC, correct_loss_cosine / num_non_zero_triplets_cosine, rtol=rtol
            )
        )
        self.assertTrue(
            torch.isclose(lossD, correct_loss_cosine / len(triplets), rtol=rtol)
        )
        self.assertTrue(
            torch.isclose(lossE, correct_smooth_loss / len(triplets), rtol=rtol)
        )

    def test_with_no_valid_triplets(self):
        loss_funcA = TripletMarginLoss(margin=0.2)
        loss_funcB = TripletMarginLoss(margin=0.2, reducer=MeanReducer())
        for dtype in TEST_DTYPES:
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.LongTensor([0, 1, 2, 3, 4])
            lossA = loss_funcA(embeddings, labels)
            lossB = loss_funcB(embeddings, labels)
            self.assertEqual(lossA, 0)
            self.assertEqual(lossB, 0)

    def test_backward(self):
        margin = 0.2
        loss_funcA = TripletMarginLoss(margin=margin)
        loss_funcB = TripletMarginLoss(margin=margin, reducer=MeanReducer())
        loss_funcC = TripletMarginLoss(smooth_loss=True)
        for dtype in TEST_DTYPES:
            for loss_func in [loss_funcA, loss_funcB, loss_funcC]:
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
