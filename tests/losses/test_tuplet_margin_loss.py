import unittest
import torch
import numpy as np
from pytorch_metric_learning.losses import TupletMarginLoss
from pytorch_metric_learning.utils import common_functions as c_f

class TestTupletMarginLoss(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.device = torch.device('cuda')

    def test_tuplet_margin_loss(self):
        margin, scale = 5, 64
        loss_func = TupletMarginLoss(margin=margin, scale=scale)

        for dtype in [torch.float16, torch.float32, torch.float64]:
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor([c_f.angle_to_coord(a) for a in embedding_angles], requires_grad=True, dtype=dtype).to(self.device) #2D embeddings
            labels = torch.LongTensor([0, 0, 1, 1, 2])

            loss = loss_func(embeddings, labels)
            loss.backward()

            pos_pairs = [(0,1), (1,0), (2,3), (3,2)]
            neg_pairs = [(0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,0), (2,1), (2,4), (3,0), (3,1), (3,4), (4,0), (4,1), (4,2), (4,3)]

            correct_total = 0
            
            for a1,p in pos_pairs:
                curr_loss = 0
                anchor1, positive = embeddings[a1], embeddings[p]
                ap_angle = torch.acos(torch.matmul(anchor1, positive)) #embeddings are normalized, so dot product == cosine
                ap_cos = torch.cos(ap_angle-np.radians(margin))
                for a2,n in neg_pairs:
                    if a2 == a1:
                        anchor2, negative = embeddings[a2], embeddings[n]
                        an_cos = torch.matmul(anchor2, negative)
                        curr_loss += torch.exp(scale*(an_cos-ap_cos))

                curr_total = torch.log(1+curr_loss)
                correct_total += curr_total
                
            correct_total /= len(pos_pairs)
            rtol = 1e-2 if dtype == torch.float16 else 1e-5
            self.assertTrue(torch.isclose(loss, correct_total, rtol=rtol))


    def test_with_no_valid_pairs(self):
        loss_func = TupletMarginLoss(0.1, 64)
        for dtype in [torch.float16, torch.float32, torch.float64]:
            embedding_angles = [0]
            embeddings = torch.tensor([c_f.angle_to_coord(a) for a in embedding_angles], requires_grad=True, dtype=dtype).to(self.device) #2D embeddings
            labels = torch.LongTensor([0])
            loss = loss_func(embeddings, labels)
            loss.backward()
            self.assertEqual(loss, 0)