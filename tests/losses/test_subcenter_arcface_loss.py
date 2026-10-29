import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_metric_learning.losses import ArcFaceLoss, SubCenterArcFaceLoss
from pytorch_metric_learning.types import SubCenterOutliers

from .. import TEST_DEVICE, TEST_DTYPES, helpers


class TestSubCenterArcFaceLoss(unittest.TestCase):
    def test_subcenter_arcface_loss(self):
        batch_size = 64
        embedding_size = 32
        margin = np.radians(30)
        scale = 64
        num_classes = 10
        for dtype in TEST_DTYPES:
            for sub_centers in [1, 3, 4]:
                loss_func = SubCenterArcFaceLoss(
                    margin=margin,
                    scale=scale,
                    num_classes=num_classes,
                    embedding_size=embedding_size,
                    sub_centers=sub_centers,
                ).to(TEST_DEVICE, dtype)

                embeddings = (
                    torch.randn(batch_size, embedding_size).to(TEST_DEVICE).type(dtype)
                )
                labels = torch.randint(low=0, high=num_classes, size=(batch_size,)).to(
                    TEST_DEVICE
                )
                self.assertTrue(loss_func.weight.shape[0] == num_classes * sub_centers)

                loss = loss_func(embeddings, labels)
                loss.backward()

                weights = F.normalize(
                    loss_func.weight.detach().double(), p=2, dim=1
                ).t()
                cosines = torch.matmul(
                    F.normalize(embeddings.detach().double()), weights
                )
                cosines, _ = cosines.view(-1, num_classes, sub_centers).max(dim=2)
                logits = cosines.clone()

                for i, c in enumerate(labels):
                    acos = torch.acos(torch.clamp(logits[i, c], -1, 1))
                    logits[i, c] = torch.cos(acos + margin)

                correct_loss = F.cross_entropy(logits * scale, labels)
                helpers.assert_close(loss, correct_loss, dtype, atol=1e-8)

                if sub_centers == 1:
                    regular_arcface = ArcFaceLoss(
                        margin=margin,
                        scale=scale,
                        num_classes=num_classes,
                        embedding_size=embedding_size,
                    ).to(TEST_DEVICE, dtype)
                    regular_arcface.weight = loss_func.weight
                    regular_arcface_loss = regular_arcface(embeddings, labels)
                    self.assertTrue(
                        torch.isclose(
                            loss, regular_arcface_loss, rtol=helpers.rtol(dtype)
                        )
                    )

                logits_out = loss_func.get_logits(embeddings)
                self.assertTrue(
                    logits_out.shape == torch.Size([batch_size, num_classes])
                )
                helpers.assert_close(logits_out / scale, cosines, dtype)

    def test_inference_subcenter_arcface(self):
        batch_size = 64
        embedding_size = 32
        margin = np.radians(30)
        scale = 64
        num_classes = 10
        sub_centers = 3
        for dtype in TEST_DTYPES:
            for threshold in [75, 90, 180]:
                loss_func = (
                    SubCenterArcFaceLoss(
                        margin=margin,
                        scale=scale,
                        num_classes=num_classes,
                        embedding_size=embedding_size,
                        sub_centers=sub_centers,
                    )
                    .to(TEST_DEVICE, dtype)
                    .to(TEST_DEVICE)
                )
                embeddings = (
                    torch.randn(batch_size, embedding_size).to(TEST_DEVICE).type(dtype)
                )
                labels = torch.randint(low=0, high=num_classes, size=(batch_size,)).to(
                    TEST_DEVICE
                )

                result = loss_func.get_outliers(
                    embeddings, labels, threshold_radians=math.radians(threshold)
                )
                self.assertIsInstance(result, SubCenterOutliers)
                outliers, dominant_centers = result.outliers, result.dominant_centers

                if threshold == 180:
                    self.assertTrue(len(outliers) == 0)
                    continue
                self.assertTrue(len(outliers) < len(labels))
                self.assertTrue(
                    dominant_centers.shape == torch.Size([num_classes, embedding_size])
                )

                cos_threshold = math.cos(math.pi * threshold / 180.0)
                distances = torch.mm(
                    F.normalize(embeddings), F.normalize(dominant_centers, dim=1).t()
                )
                outliers_labels = labels[outliers]
                outliers_distances = distances[outliers, outliers_labels]
                self.assertTrue((outliers_distances < cos_threshold).all())

                all_indices = torch.arange(len(labels), device=TEST_DEVICE)
                normal_indices = torch.masked_select(
                    all_indices, distances[all_indices, labels] >= cos_threshold
                )
                self.assertTrue(
                    normal_indices.shape[0] + outliers.shape[0] == labels.shape[0]
                )
                self.assertTrue(
                    len(
                        np.intersect1d(
                            normal_indices.cpu().numpy(), outliers.cpu().numpy()
                        )
                    )
                    == 0
                )

    def test_reducer_subcenter_arcface(self):

        arcfaceloss = SubCenterArcFaceLoss(
            num_classes=10,
            sub_centers=3,
            embedding_size=64,
        ).to(TEST_DEVICE)

        emb = torch.randn(4, 64, device=TEST_DEVICE)
        result = arcfaceloss.loss_terms(emb, torch.arange(4, device=TEST_DEVICE))

        self.assertTrue(isinstance(result, dict))
        self.assertTrue(result["loss"]["losses"].shape[0] == 4)
