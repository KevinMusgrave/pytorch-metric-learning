import unittest
from itertools import chain

import torch

from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import CentroidTripletLoss
from pytorch_metric_learning.reducers import MeanReducer

from .. import TEST_DEVICE, TEST_DTYPES
from ..zzz_testing_utils.testing_utils import angle_to_coord


def normalize(embeddings):
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=0)
    return embeddings


class TestCentroidTripletLoss(unittest.TestCase):
    def test_indices_tuple_failure(self):
        loss_fn = CentroidTripletLoss()
        for labels_arr in [[0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 1, 1, 2, 2, 3]]:
            embeddings = torch.randn(len(labels_arr), 32, device=TEST_DEVICE)
            labels = torch.tensor(labels_arr, device=TEST_DEVICE)
            with self.assertRaises(ValueError) as context:
                loss_fn(embeddings, labels)

    def test_centroid_triplet_loss(self):
        for dtype in TEST_DTYPES:
            embedding_angles = [0, 10, 20, 30, 40, 50]
            centroid_makers = [
                [[10, 20], [30, 40, 50]],
                [[0, 20], [30, 40, 50]],
                [[0, 10], [30, 40, 50]],
                [[40, 50], [0, 10, 20]],
                [[30, 50], [0, 10, 20]],
                [[30, 40], [0, 10, 20]],
            ]
            triplets = [
                (0, (0, 0), (0, 1)),
                (1, (1, 0), (1, 1)),
                (2, (2, 0), (2, 1)),
                (3, (3, 0), (3, 1)),
                (4, (4, 0), (4, 1)),
                (5, (5, 0), (5, 1)),
            ]

            labels = torch.LongTensor([0, 0, 0, 1, 1, 1])

            self.helper(embedding_angles, centroid_makers, labels, triplets, dtype)

    def test_sorting_invariance(self):
        for dtype in TEST_DTYPES:
            centroid_makers = [
                [[10, 20], [30, 40, 50]],
                [[40, 50], [0, 10, 20]],
                [[30, 50], [0, 10, 20]],
                [[0, 20], [30, 40, 50]],
                [[0, 10], [30, 40, 50]],
                [[30, 40], [0, 10, 20]],
            ]

            embedding_angles = [0, 30, 40, 10, 20, 50]
            labels = torch.LongTensor([0, 1, 1, 0, 0, 1])

            triplets = [
                (0, (0, 0), (0, 1)),
                (1, (1, 0), (1, 1)),
                (2, (2, 0), (2, 1)),
                (3, (3, 0), (3, 1)),
                (4, (4, 0), (4, 1)),
                (5, (5, 0), (5, 1)),
            ]

            self.helper(embedding_angles, centroid_makers, labels, triplets, dtype)

    def test_imbalanced(self):
        for dtype in TEST_DTYPES:
            per_class_angles = [
                [0, 10, 20],  # class A
                [30, 40, 50, 55],  # class B
                [60, 70, 80, 90],  # class C
            ]
            embedding_angles = chain(*per_class_angles)
            labels = torch.LongTensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

            centroid_makers = [
                [[10, 20], [30, 40, 50, 55]],
                [[10, 20], [60, 70, 80, 90]],
                [[0, 20], [30, 40, 50, 55]],
                [[0, 20], [60, 70, 80, 90]],
                [[0, 10], [30, 40, 50, 55]],
                [[0, 10], [60, 70, 80, 90]],
                [[40, 50, 55], [0, 10, 20]],
                [[40, 50, 55], [60, 70, 80, 90]],
                [[30, 50, 55], [0, 10, 20]],
                [[30, 50, 55], [60, 70, 80, 90]],
                [[30, 40, 55], [0, 10, 20]],
                [[30, 40, 55], [60, 70, 80, 90]],
                [[30, 40, 50], [0, 10, 20]],
                [[30, 40, 50], [60, 70, 80, 90]],
                [[70, 80, 90], [0, 10, 20]],
                [[70, 80, 90], [30, 40, 50, 55]],
                [[60, 80, 90], [0, 10, 20]],
                [[60, 80, 90], [30, 40, 50, 55]],
                [[60, 70, 90], [0, 10, 20]],
                [[60, 70, 90], [30, 40, 50, 55]],
                [[60, 70, 80], [0, 10, 20]],
                [[60, 70, 80], [30, 40, 50, 55]],
            ]

            triplets = [
                (0, (0, 0), (0, 1)),
                (0, (1, 0), (1, 1)),
                (1, (2, 0), (2, 1)),
                (1, (3, 0), (3, 1)),
                (2, (4, 0), (4, 1)),
                (2, (5, 0), (5, 1)),
                (3, (6, 0), (6, 1)),
                (3, (7, 0), (7, 1)),
                (4, (8, 0), (8, 1)),
                (4, (9, 0), (9, 1)),
                (5, (10, 0), (10, 1)),
                (5, (11, 0), (11, 1)),
                (6, (12, 0), (12, 1)),
                (6, (13, 0), (13, 1)),
                (7, (14, 0), (14, 1)),
                (7, (15, 0), (15, 1)),
                (8, (16, 0), (16, 1)),
                (8, (17, 0), (17, 1)),
                (9, (18, 0), (18, 1)),
                (9, (19, 0), (19, 1)),
                (10, (20, 0), (20, 1)),
                (10, (21, 0), (21, 1)),
            ]

            self.helper(embedding_angles, centroid_makers, labels, triplets, dtype)

    def helper(
        self,
        embedding_angles,
        centroid_makers,
        labels,
        triplets,
        dtype,
        ref_emb=None,
        ref_labels=None,
    ):
        embeddings = torch.tensor(
            [angle_to_coord(a) for a in embedding_angles],
            requires_grad=True,
            dtype=dtype,
        ).to(
            TEST_DEVICE
        )  # 2D embeddings

        centroids = [
            [
                torch.stack(
                    [
                        torch.tensor(
                            angle_to_coord(a), requires_grad=True, dtype=dtype
                        ).to(TEST_DEVICE)
                        for a in coords
                    ]
                ).mean(-2)
                for coords in one_maker
            ]
            for one_maker in centroid_makers
        ]

        margin = 0.2
        loss_funcA = CentroidTripletLoss(margin=margin)
        loss_funcB = CentroidTripletLoss(margin=margin, reducer=MeanReducer())
        loss_funcC = CentroidTripletLoss(margin=margin, distance=CosineSimilarity())
        loss_funcD = CentroidTripletLoss(
            margin=margin, reducer=MeanReducer(), distance=CosineSimilarity()
        )
        loss_funcE = CentroidTripletLoss(margin=margin, smooth_loss=True)

        [lossA, lossB, lossC, lossD, lossE] = [
            x(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
            for x in [loss_funcA, loss_funcB, loss_funcC, loss_funcD, loss_funcE]
        ]

        correct_loss = 0
        correct_loss_cosine = 0
        correct_smooth_loss = 0
        num_non_zero_triplets = 0
        num_non_zero_triplets_cosine = 0

        for a, pc, nc in triplets:
            anchor = embeddings[a]

            positive = centroids[pc[0]][pc[1]]
            negative = centroids[nc[0]][nc[1]]

            anchor = normalize(anchor)
            positive = normalize(positive)
            negative = normalize(negative)

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

    def test_backward(self):
        margin = 0.2
        loss_funcA = CentroidTripletLoss(margin=margin)
        loss_funcB = CentroidTripletLoss(margin=margin, reducer=MeanReducer())
        loss_funcC = CentroidTripletLoss(smooth_loss=True)
        for dtype in TEST_DTYPES:
            for loss_func in [loss_funcA, loss_funcB, loss_funcC]:
                embedding_angles = [0, 20, 40, 60, 80, 85]
                embeddings = torch.tensor(
                    [angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(
                    TEST_DEVICE
                )  # 2D embeddings
                labels = torch.LongTensor([0, 0, 1, 1, 2, 2])

                loss = loss_func(embeddings, labels)
                loss.backward()
