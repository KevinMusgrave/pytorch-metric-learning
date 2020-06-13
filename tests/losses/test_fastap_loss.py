
######################################
#######ORIGINAL IMPLEMENTATION########
######################################
# FROM https://github.com/kunhe/FastAP-metric-learning/blob/master/pytorch/FastAP_loss.py
# This code is copied directly from the official implementation
# so that we can make sure our implementation returns the same result.
# It's copied under the MIT license.
import torch
from torch.autograd import Variable, Function

def softBinning(D, mid, Delta):
    y = 1 - torch.abs(D-mid)/Delta
    return torch.max(torch.Tensor([0]).cuda(), y)

def dSoftBinning(D, mid, Delta):
    side1 = (D > (mid - Delta)).type(torch.float)
    side2 = (D <= mid).type(torch.float)
    ind1 = (side1 * side2) #.type(torch.uint8)

    side1 = (D > mid).type(torch.float)
    side2 = (D <= (mid + Delta)).type(torch.float)
    ind2 = (side1 * side2) #.type(torch.uint8)

    return (ind1 - ind2)/Delta
    
######################################
#######ORIGINAL IMPLEMENTATION########
######################################
# FROM https://github.com/kunhe/FastAP-metric-learning/blob/master/pytorch/FastAP_loss.py
# This code is copied directly from the official implementation
# so that we can make sure our implementation returns the same result.
# It's copied under the MIT license.
class OriginalImplementationFastAP(torch.autograd.Function):
    """
    FastAP - autograd function definition

    This class implements the FastAP loss from the following paper:
    "Deep Metric Learning to Rank", 
    F. Cakir, K. He, X. Xia, B. Kulis, S. Sclaroff. CVPR 2019

    NOTE:
        Given a input batch, FastAP does not sample triplets from it as it's not 
        a triplet-based method. Therefore, FastAP does not take a Sampler as input. 
        Rather, we specify how the input batch is selected.
    """

    @staticmethod
    def forward(ctx, input, target, num_bins):
        """
        Args:
            input:     torch.Tensor(N x embed_dim), embedding matrix
            target:    torch.Tensor(N x 1), class labels
            num_bins:  int, number of bins in distance histogram
        """
        N = target.size()[0]
        assert input.size()[0] == N, "Batch size donesn't match!"
        
        # 1. get affinity matrix
        Y   = target.unsqueeze(1)
        Aff = 2 * (Y == Y.t()).type(torch.float) - 1
        Aff.masked_fill_(torch.eye(N, N).bool().cuda(), 0)  # set diagonal to 0

        I_pos = (Aff > 0).type(torch.float).cuda()
        I_neg = (Aff < 0).type(torch.float).cuda()
        N_pos = torch.sum(I_pos, 1)

        # 2. compute distances from embeddings
        # squared Euclidean distance with range [0,4]
        dist2 = 2 - 2 * torch.mm(input, input.t())

        # 3. estimate discrete histograms
        Delta = torch.tensor(4. / num_bins).cuda()
        Z     = torch.linspace(0., 4., steps=num_bins+1).cuda()
        L     = Z.size()[0]
        h_pos = torch.zeros((N, L)).cuda()
        h_neg = torch.zeros((N, L)).cuda()
        for l in range(L):
            pulse    = softBinning(dist2, Z[l], Delta)
            h_pos[:,l] = torch.sum(pulse * I_pos, 1)
            h_neg[:,l] = torch.sum(pulse * I_neg, 1)

        H_pos = torch.cumsum(h_pos, 1)
        h     = h_pos + h_neg
        H     = torch.cumsum(h, 1)
        
        # 4. compate FastAP
        FastAP = h_pos * H_pos / H
        FastAP[torch.isnan(FastAP) | torch.isinf(FastAP)] = 0
        FastAP = torch.sum(FastAP,1)/N_pos
        FastAP = FastAP[ ~torch.isnan(FastAP) ]
        loss   = 1 - torch.mean(FastAP)
        if torch.rand(1) > 0.99:
            print("loss value (1-mean(FastAP)): ", loss.item())

        # 6. save for backward
        ctx.save_for_backward(input, target)
        ctx.Z     = Z
        ctx.Delta = Delta
        ctx.dist2 = dist2
        ctx.I_pos = I_pos
        ctx.I_neg = I_neg
        ctx.h_pos = h_pos
        ctx.h_neg = h_neg
        ctx.H_pos = H_pos
        ctx.N_pos = N_pos
        ctx.h     = h
        ctx.H     = H
        ctx.L     = torch.tensor(L)
        
        return loss

    
    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors

        Z     = Variable(ctx.Z     , requires_grad = False)
        Delta = Variable(ctx.Delta , requires_grad = False)
        dist2 = Variable(ctx.dist2 , requires_grad = False)
        I_pos = Variable(ctx.I_pos , requires_grad = False)
        I_neg = Variable(ctx.I_neg , requires_grad = False)
        h     = Variable(ctx.h     , requires_grad = False)
        H     = Variable(ctx.H     , requires_grad = False)
        h_pos = Variable(ctx.h_pos , requires_grad = False)
        h_neg = Variable(ctx.h_neg , requires_grad = False)
        H_pos = Variable(ctx.H_pos , requires_grad = False)
        N_pos = Variable(ctx.N_pos , requires_grad = False)

        L     = Z.size()[0]
        H2    = torch.pow(H,2)
        H_neg = H - H_pos

        # 1. d(FastAP)/d(h+)
        LTM1 = torch.tril(torch.ones(L,L), -1)  # lower traingular matrix
        tmp1 = h_pos * H_neg / H2
        tmp1[torch.isnan(tmp1)] = 0

        d_AP_h_pos = (H_pos * H + h_pos * H_neg) / H2 
        d_AP_h_pos = d_AP_h_pos + torch.mm(tmp1, LTM1.cuda())
        d_AP_h_pos = d_AP_h_pos / N_pos.repeat(L,1).t()
        d_AP_h_pos[torch.isnan(d_AP_h_pos) | torch.isinf(d_AP_h_pos)] = 0


        # 2. d(FastAP)/d(h-)
        LTM0 = torch.tril(torch.ones(L,L), 0)  # lower triangular matrix
        tmp2 = -h_pos * H_pos / H2
        tmp2[torch.isnan(tmp2)] = 0

        d_AP_h_neg = torch.mm(tmp2, LTM0.cuda())
        d_AP_h_neg = d_AP_h_neg / N_pos.repeat(L,1).t()
        d_AP_h_neg[torch.isnan(d_AP_h_neg) | torch.isinf(d_AP_h_neg)] = 0


        # 3. d(FastAP)/d(embedding)
        d_AP_x = 0
        for l in range(L):
            dpulse = dSoftBinning(dist2, Z[l], Delta)
            dpulse[torch.isnan(dpulse) | torch.isinf(dpulse)] = 0
            ddp = dpulse * I_pos
            ddn = dpulse * I_neg

            alpha_p = torch.diag(d_AP_h_pos[:,l]) # N*N
            alpha_n = torch.diag(d_AP_h_neg[:,l])
            Ap = torch.mm(ddp, alpha_p) + torch.mm(alpha_p, ddp)
            An = torch.mm(ddn, alpha_n) + torch.mm(alpha_n, ddn)

            # accumulate gradient 
            d_AP_x = d_AP_x - torch.mm(input.t(), (Ap+An))

        grad_input = -d_AP_x
        return grad_input.t(), None, None    

######################################
#######ORIGINAL IMPLEMENTATION########
######################################
# FROM https://github.com/kunhe/FastAP-metric-learning/blob/master/pytorch/FastAP_loss.py
# This code is copied directly from the official implementation
# so that we can make sure our implementation returns the same result.
# It's copied under the MIT license.
class OriginalImplementationFastAPLoss(torch.nn.Module):
    """
    FastAP - loss layer definition

    This class implements the FastAP loss from the following paper:
    "Deep Metric Learning to Rank", 
    F. Cakir, K. He, X. Xia, B. Kulis, S. Sclaroff. CVPR 2019
    """
    def __init__(self, num_bins=10):
        super(OriginalImplementationFastAPLoss, self).__init__()
        self.num_bins = num_bins

    def forward(self, batch, labels):
        return OriginalImplementationFastAP.apply(batch, labels, self.num_bins)



### Testing this library's implementation ###
import unittest
import numpy as np
from pytorch_metric_learning.losses import FastAPLoss
from pytorch_metric_learning.utils import common_functions as c_f

class TestCosFaceLoss(unittest.TestCase):
    def test_cosface_loss(self):
        num_bins = 5
        loss_func = FastAPLoss(num_bins)
        original_loss_func = OriginalImplementationFastAPLoss(num_bins)
        device = torch.device("cuda")

        embedding_angles = torch.arange(0, 180)
        embeddings = torch.tensor([c_f.angle_to_coord(a) for a in embedding_angles], requires_grad=True, dtype=torch.float).to(device) #2D embeddings
        labels = torch.randint(low=0, high=10, size=(180,)).to(device)

        loss = loss_func(embeddings, labels)
        loss.backward()
        original_loss = original_loss_func(embeddings, labels)
        self.assertTrue(torch.isclose(loss, original_loss))