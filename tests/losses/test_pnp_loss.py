import unittest

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from pytorch_metric_learning.losses import PNPLoss
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from .. import TEST_DEVICE, TEST_DTYPES
from ..zzz_testing_utils.testing_utils import angle_to_coord


class OriginalImplementationPNP(nn.Module):
    def __init__(self, b, alpha, anneal, variant, **kwargs):
        super(OriginalImplementationPNP, self).__init__()
        self.b = b
        self.alpha = alpha
        self.anneal = anneal
        self.variant = variant
        
    def forward(self, batch, labels, **kwargs):
        #if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()
        # compute the relevance scores via cosine similarity of the CNN-produced embedding vectors
        dtype, device = batch.dtype, batch.device

        N = labels.size(0)
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels)
        I_pos = torch.zeros(N, N, dtype=dtype, device=device)
        I_neg = torch.zeros(N, N, dtype=dtype, device=device)
        I_pos[a1_idx, p_idx] = 1
        I_neg[a2_idx, n_idx] = 1
        N_pos = torch.sum(I_pos, dim=1)
        safe_N = N_pos > 0
        if torch.sum(safe_N) == 0:
            return 0
        self.mask = I_neg.unsqueeze(dim=1).repeat(1, N, 1)
        
        sim_all = self.compute_aff(batch)
        sim_all = sim_all 
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, N, 1)
        # compute the difference matrix
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        # pass through the sigmoid and ignores the relevance score of the query to itself
        sim_sg = self.sigmoid(sim_diff, temp=self.anneal) * self.mask.cuda()
        # compute the rankings,all batch
        sim_all_rk = torch.sum(sim_sg, dim=-1) 
        if self.variant == 'PNP-D_s':
            sim_all_rk = torch.log(1+sim_all_rk)
        elif self.variant == 'PNP-D_q':
            sim_all_rk = 1/(1+sim_all_rk)**(self.alpha)


        elif self.variant == 'PNP-I_u':
            sim_all_rk = (1+sim_all_rk)*torch.log(1+sim_all_rk)
            
        elif self.variant == 'PNP-I_b':
            b = self.b
            sim_all_rk = 1/b**2 * (b*sim_all_rk-torch.log(1+b*sim_all_rk))
        elif self.variant == 'PNP-O':
            pass
        else:
                raise Exception('variantation <{}> not available!'.format(self.variant))
        
        
        loss = (sim_all_rk * I_pos.cuda()) / N_pos.reshape(-1, 1)
        loss = torch.sum(loss) / N
        if  self.variant == 'PNP-D_q':
            return 1 - loss
        else:
            return loss


    def sigmoid(self, tensor, temp=1.0):
        """ temperature controlled sigmoid
        takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
        """
        exponent = -tensor / temp
        # clamp the input tensor for stability
        exponent = torch.clamp(exponent, min=-50, max=50)
        y = 1.0 / (1.0 + torch.exp(exponent))
        return y


    def compute_aff(self, x):
        """computes the affinity matrix between an input vector and itself"""
        return torch.mm(x, x.t())


class TestFastAPLoss(unittest.TestCase):
    def test_fast_ap_loss(self):
        b, alpha, anneal, variant = 2, 4, 0.01, "PNP-D_q"
        loss_func = PNPLoss(b, alpha, anneal, variant)
        original_loss_func = OriginalImplementationPNP( b, alpha, anneal, variant).to(TEST_DEVICE)


        for dtype in TEST_DTYPES:
            embedding_angles = torch.arange(0, 180)
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.randint(low=0, high=10, size=(180,)).to(TEST_DEVICE)

            loss = loss_func(embeddings, labels)
            loss.backward()

            correct_loss = original_loss_func(F.normalize(embeddings), labels)
            self.assertTrue(torch.isclose(loss, correct_loss))

