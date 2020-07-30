import unittest
import torch
from pytorch_metric_learning.losses import NormalizedSoftmaxLoss
from pytorch_metric_learning.utils import common_functions as c_f

class TestNormalizedSoftmaxLoss(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.device = torch.device('cuda')

    def test_normalized_softmax_loss(self):
        temperature = 0.1
        for dtype in [torch.float16, torch.float32, torch.float64]:
            loss_func = NormalizedSoftmaxLoss(temperature=temperature, num_classes=10, embedding_size=2).to(self.device)
            embedding_angles = torch.arange(0, 180)
            embeddings = torch.tensor([c_f.angle_to_coord(a) for a in embedding_angles], requires_grad=True, dtype=dtype).to(self.device) #2D embeddings
            labels = torch.randint(low=0, high=10, size=(180,)).to(self.device)

            loss = loss_func(embeddings, labels)
            loss.backward()

            weights = torch.nn.functional.normalize(loss_func.W, p=2, dim=0)
            logits = torch.matmul(embeddings, weights)
            correct_loss = torch.nn.functional.cross_entropy(logits/temperature, labels)
            self.assertTrue(torch.isclose(loss, correct_loss))