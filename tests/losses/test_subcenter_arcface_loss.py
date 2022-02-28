import unittest
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import SubCenterArcFaceLoss, ArcFaceLoss
import math

from .. import TEST_DEVICE, TEST_DTYPES


class TestSubCenterArcFaceLoss(unittest.TestCase):
    def test_subcenter_arcface_loss(self):
        batch_size = 64
        embedding_size = 32
        margin = 30
        scale = 64
        num_classes = 10
        for dtype in TEST_DTYPES:
            for sub_centers in [1, 3, 4]:
                loss_func = SubCenterArcFaceLoss(
                    margin=margin, scale=scale, num_classes=num_classes, embedding_size=embedding_size, sub_centers=sub_centers
                )

                embeddings = torch.randn(batch_size, embedding_size).to(TEST_DEVICE).type(dtype)
                labels = torch.randint(low=0, high=num_classes, size=(batch_size,)).to(TEST_DEVICE)
                # check if subcenters are included
                self.assertTrue(loss_func.W.shape[1] == num_classes * sub_centers)
                
                loss = loss_func(embeddings, labels)
                loss.backward()

                weights = F.normalize(loss_func.W, p=2, dim=0)
                logits = torch.matmul(F.normalize(embeddings), weights)
                # include only closest sub centers
                logits = logits.view(-1, num_classes, sub_centers)
                logits, _ = logits.max(axis=2)

                for i, c in enumerate(labels):
                    acos = torch.acos(torch.clamp(logits[i, c], -1, 1))
                    logits[i, c] = torch.cos(
                        acos + torch.tensor(np.radians(margin), dtype=dtype).to(TEST_DEVICE)
                    )
        
                correct_loss = F.cross_entropy(logits * scale, labels.to(TEST_DEVICE))
        
                rtol = 1e-2 if dtype == torch.float16 else 1e-5
                self.assertTrue(torch.isclose(loss, correct_loss, rtol=rtol))

                if sub_centers == 1:
                    regular_arcface = ArcFaceLoss(margin=margin, scale=scale, num_classes=num_classes, embedding_size=embedding_size)
                    regular_arcface.W = loss_func.W
                    regular_arcface_loss = regular_arcface(embeddings, labels)
                    self.assertTrue(torch.isclose(loss, regular_arcface_loss, rtol=rtol))
                
                # test get_logits
                logits_out = loss_func.get_logits(embeddings)
                self.assertTrue(logits_out.shape == torch.Size([batch_size, num_classes]))
                logits = torch.matmul(F.normalize(embeddings), weights)
                # include only closest sub centers
                logits = logits.view(-1, num_classes, sub_centers)
                logits_target, _ = logits.max(axis=2)
                self.assertTrue(
                    torch.allclose(
                        logits_out, logits_target * scale
                    )
                )
    
    def test_inference_subcenter_arcface(self):
        batch_size = 64
        embedding_size = 32
        margin = 30
        scale = 64
        num_classes = 10
        sub_centers = 3
        threshold = 75
        for dtype in TEST_DTYPES:
            loss_func = SubCenterArcFaceLoss(
                margin=margin, scale=scale, num_classes=num_classes, embedding_size=embedding_size, sub_centers=sub_centers
            ).to(
                TEST_DEVICE
            )
            embeddings = torch.randn(batch_size, embedding_size).to(TEST_DEVICE).type(dtype)
            labels = torch.randint(low=0, high=num_classes, size=(batch_size,)).to(TEST_DEVICE)

            outliers, dominant_centers = loss_func.get_outliers(embeddings, labels, threshold=threshold)
                        
            self.assertTrue(len(outliers) < len(labels))
            
            self.assertTrue(dominant_centers.shape[1] == num_classes)
                
            cos_threshold = math.cos(math.pi * threshold / 180.)
            distances = torch.mm(F.normalize(embeddings), dominant_centers)
            outliers_labels = labels[outliers]
            outliers_distances = distances[outliers, outliers_labels]
            # check of outliers are below the threshold
            self.assertTrue((outliers_distances < cos_threshold).all())
            
            all_indices = torch.arange(len(labels), device=TEST_DEVICE)
            normal_indices = torch.masked_select(all_indices, distances[all_indices, labels] >= cos_threshold)
            # check if all indeces present
            self.assertTrue((normal_indices.shape[0] + outliers.shape[0] == labels.shape[0]))
            # check if there's no intersection between indeces of 2 sets            
            self.assertTrue(len(np.intersect1d(normal_indices.cpu().numpy(), outliers.cpu().numpy())) == 0)
