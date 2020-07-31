import unittest 
from .. import TEST_DTYPES
import torch
from pytorch_metric_learning.losses import ProxyNCALoss
from pytorch_metric_learning.utils import common_functions as c_f

class TestProxyNCALoss(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.device = torch.device('cuda')

    def test_proxy_nca_loss(self):
        for dtype in TEST_DTYPES:
            softmax_scale = 1 if dtype == torch.float16 else 10
            loss_func = ProxyNCALoss(softmax_scale=softmax_scale, num_classes=10, embedding_size=2)

            embedding_angles = torch.arange(0, 180)
            embeddings = torch.tensor([c_f.angle_to_coord(a) for a in embedding_angles], requires_grad=True, dtype=dtype).to(self.device) #2D embeddings
            labels = torch.randint(low=0, high=10, size=(180,))

            loss = loss_func(embeddings, labels)
            loss.backward()

            proxies = torch.nn.functional.normalize(loss_func.proxies, p=2, dim=1)
            correct_loss = 0
            for i in range(len(embeddings)):
                curr_emb, curr_label = embeddings[i], labels[i]
                curr_proxy = proxies[curr_label]
                denominator = torch.sum((curr_emb-proxies)**2, dim=1)
                denominator = torch.sum(torch.exp(-denominator*softmax_scale))
                numerator = torch.sum((curr_emb-curr_proxy)**2)
                numerator = torch.exp(-numerator*softmax_scale)
                correct_loss += -torch.log(numerator/denominator)
            
            correct_loss /= len(embeddings)
            rtol = 1e-2 if dtype == torch.float16 else 1e-5
            self.assertTrue(torch.isclose(loss, correct_loss, rtol=rtol))