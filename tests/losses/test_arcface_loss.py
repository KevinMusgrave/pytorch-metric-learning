import unittest
import torch
import numpy as np
from pytorch_metric_learning.losses import ArcFaceLoss
from pytorch_metric_learning.utils import common_functions as c_f

class TestArcFaceLoss(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.device = torch.device('cuda')

    def test_arcface_loss(self):
        margin = 30
        scale = 64

        for dtype in [torch.float16, torch.float32, torch.float64]:
            loss_func = ArcFaceLoss(margin=margin, scale=scale, num_classes=10, embedding_size=2).to(self.device)
            embedding_angles = torch.arange(0, 180)
            embeddings = torch.tensor([c_f.angle_to_coord(a) for a in embedding_angles], requires_grad=True, dtype=dtype).to(self.device) #2D embeddings
            labels = torch.randint(low=0, high=10, size=(180,))

            loss = loss_func(embeddings, labels)
            loss.backward()

            weights = torch.nn.functional.normalize(loss_func.W, p=2, dim=0)
            logits = torch.matmul(embeddings, weights)
            for i, c in enumerate(labels):
                logits[i, c] = torch.cos(torch.acos(logits[i, c]) + torch.tensor(np.radians(margin), dtype=dtype).to(self.device))
            
            correct_loss = torch.nn.functional.cross_entropy(logits*scale, labels.to(self.device))

            rtol = 1e-2 if dtype == torch.float16 else 1e-5
            self.assertTrue(torch.isclose(loss, correct_loss, rtol=rtol))