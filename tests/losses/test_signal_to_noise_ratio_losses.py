import unittest
import torch
from pytorch_metric_learning.losses import SignalToNoiseRatioContrastiveLoss
from pytorch_metric_learning.utils import common_functions as c_f

class TestSNRContrastiveLoss(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.device = torch.device('cuda')

    def test_snr_contrastive_loss(self):
        pos_margin, neg_margin, regularizer_weight = 0, 0.1, 0.1
        loss_func = SignalToNoiseRatioContrastiveLoss(pos_margin=pos_margin, neg_margin=neg_margin, regularizer_weight=regularizer_weight)

        for dtype in [torch.float16, torch.float32, torch.float64]:
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor([c_f.angle_to_coord(a) for a in embedding_angles], requires_grad=True, dtype=dtype).to(self.device) #2D embeddings
            labels = torch.LongTensor([0, 0, 1, 1, 2])

            loss = loss_func(embeddings, labels)
            loss.backward()

            pos_pairs = [(0,1), (1,0), (2,3), (3,2)]
            neg_pairs = [(0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,0), (2,1), (2,4), (3,0), (3,1), (3,4), (4,0), (4,1), (4,2), (4,3)]

            correct_pos_loss = 0
            correct_neg_loss = 0
            num_non_zero = 0
            for a,p in pos_pairs:
                anchor, positive = embeddings[a], embeddings[p]
                curr_loss = torch.relu(torch.var(anchor-positive) / torch.var(anchor) - pos_margin)
                correct_pos_loss += curr_loss
                if curr_loss > 0:
                    num_non_zero += 1
            if num_non_zero > 0:
                correct_pos_loss /= num_non_zero

            num_non_zero = 0
            for a,n in neg_pairs:
                anchor, negative = embeddings[a], embeddings[n]
                curr_loss = torch.relu(neg_margin - torch.var(anchor-negative) / torch.var(anchor))
                correct_neg_loss += curr_loss
                if curr_loss > 0:
                    num_non_zero += 1
            if num_non_zero > 0:
                correct_neg_loss /= num_non_zero

            reg_loss = torch.mean(torch.abs(torch.sum(embeddings, dim=1)))

            correct_total = correct_pos_loss + correct_neg_loss + regularizer_weight*reg_loss
            rtol = 1e-2 if dtype == torch.float16 else 1e-5
            self.assertTrue(torch.isclose(loss, correct_total, rtol=rtol))


    def test_with_no_valid_pairs(self):
        reg_weight = 0.1
        loss_func = SignalToNoiseRatioContrastiveLoss(0, 0.5, reg_weight)
        for dtype in [torch.float16, torch.float32, torch.float64]:
            embedding_angles = [0]
            embeddings = torch.tensor([c_f.angle_to_coord(a) for a in embedding_angles], requires_grad=True, dtype=dtype).to(self.device) #2D embeddings
            labels = torch.LongTensor([0])
            reg_loss = torch.mean(torch.abs(torch.sum(embeddings, dim=1)))*reg_weight
            loss = loss_func(embeddings, labels)
            loss.backward()
            self.assertEqual(loss, reg_loss)