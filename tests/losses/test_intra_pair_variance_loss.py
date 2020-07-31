import unittest 
from .. import TEST_DTYPES
import torch
from pytorch_metric_learning.losses import IntraPairVarianceLoss
from pytorch_metric_learning.utils import common_functions as c_f

class TestIntraPairVarianceLoss(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.device = torch.device('cuda')

    def test_intra_pair_variance_loss(self):
        pos_eps, neg_eps = 0.01, 0.02
        loss_func = IntraPairVarianceLoss(pos_eps, neg_eps)

        for dtype in TEST_DTYPES:
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor([c_f.angle_to_coord(a) for a in embedding_angles], requires_grad=True, dtype=dtype).to(self.device) #2D embeddings
            labels = torch.LongTensor([0, 0, 1, 1, 2])

            loss = loss_func(embeddings, labels)
            loss.backward()

            pos_pairs = [(0,1), (1,0), (2,3), (3,2)]
            neg_pairs = [(0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,0), (2,1), (2,4), (3,0), (3,1), (3,4), (4,0), (4,1), (4,2), (4,3)]

            pos_total, neg_total = 0, 0
            mean_pos = 0
            mean_neg = 0
            for a,p in pos_pairs:
                mean_pos += torch.matmul(embeddings[a], embeddings[p])
            for a,n in neg_pairs:
                mean_neg += torch.matmul(embeddings[a], embeddings[n])
            mean_pos /= len(pos_pairs)
            mean_neg /= len(neg_pairs)

            for a,p in pos_pairs:
                pos_total += torch.relu(((1-pos_eps)*mean_pos - torch.matmul(embeddings[a], embeddings[p])))**2
            for a,n in neg_pairs:
                neg_total += torch.relu((torch.matmul(embeddings[a], embeddings[n])-(1+neg_eps)*mean_neg))**2

            pos_total /= len(pos_pairs)
            neg_total /= len(neg_pairs)
            correct_total = pos_total+neg_total
            rtol = 1e-2 if dtype == torch.float16 else 1e-5
            self.assertTrue(torch.isclose(loss, correct_total, rtol=rtol))


    def test_with_no_valid_pairs(self):
        loss_func = IntraPairVarianceLoss(0.01,0.01)
        for dtype in TEST_DTYPES:
            embedding_angles = [0]
            embeddings = torch.tensor([c_f.angle_to_coord(a) for a in embedding_angles], requires_grad=True, dtype=dtype).to(self.device) #2D embeddings
            labels = torch.LongTensor([0])
            loss = loss_func(embeddings, labels)
            loss.backward()
            self.assertEqual(loss, 0)