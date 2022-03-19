import unittest

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_metric_learning.losses import DynamicArcFaceLoss, ArcFaceLoss, SubCenterArcFaceLoss

from .. import TEST_DEVICE, TEST_DTYPES
from ..zzz_testing_utils.testing_utils import angle_to_coord


class TestDynamicArcFaceLoss(unittest.TestCase):
    def test_dynamicarcface_loss(self):
        scale = 64
        n = torch.tensor([4,5,2,3,1,2,3,4,4,5])
        sub_centers = 3
        num_classes = 10
        a = 0.5
        b = 0.05
        lambda0 = 0.25
        for loss_type  in (ArcFaceLoss, SubCenterArcFaceLoss):
            for dtype in TEST_DTYPES:
                loss_func = DynamicArcFaceLoss(
                   n, 
                   scale=scale, 
                   num_classes=10,  
                   embedding_size=2, 
                   loss_func=loss_type,
                   a=a,
                   b=b,
                   lambda0=lambda0
                )
                embedding_angles = torch.arange(0, 180)
                embeddings = torch.tensor(
                    [angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(
                    TEST_DEVICE
                )  # 2D embeddings
                labels = torch.randint(low=0, high=10, size=(180,))
    
                loss = loss_func(embeddings, labels)
                loss.backward()
    
                weights = F.normalize(loss_func.loss_func.W, p=2, dim=0)
                logits = torch.matmul(F.normalize(embeddings), weights)
                if loss_type == SubCenterArcFaceLoss:
                    logits = logits.view(-1, num_classes, sub_centers)
                    logits, _ = logits.max(axis=2)
                class_margins = a * n ** (-lambda0) + b
                batch_margins = class_margins[labels].to(dtype=dtype, device=TEST_DEVICE)

                
                for i, c in enumerate(labels):
                    acos = torch.acos(torch.clamp(logits[i, c], -1, 1))
                    logits[i, c] = torch.cos(
                        acos + batch_margins[i].to(TEST_DEVICE)
                    )
    
                correct_loss = F.cross_entropy(logits * scale, labels.to(TEST_DEVICE))

                rtol = 1e-2 if dtype == torch.float16 else 1e-5

                self.assertTrue(torch.isclose(loss, correct_loss, rtol=rtol))

