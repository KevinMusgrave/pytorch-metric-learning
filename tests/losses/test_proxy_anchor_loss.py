######################################
#######ORIGINAL IMPLEMENTATION########
######################################
# FROM https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/blob/master/code/losses.py
# This code is copied directly from the official implementation
# so that we can make sure our implementation returns the same result.
# It's copied under the MIT license.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class OriginalImplementationProxyAnchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss



import unittest
import torch
from pytorch_metric_learning.losses import ProxyAnchorLoss
from pytorch_metric_learning.utils import common_functions as c_f

class TestProxyAnchorLoss(unittest.TestCase):
    def test_proxyanchor_loss(self):
        num_classes = 10
        embedding_size = 2
        margin = 0.5
        alpha = 32
        device = torch.device("cuda")
        loss_func = ProxyAnchorLoss(num_classes, embedding_size, margin = margin, alpha = alpha).to(device)
        original_loss_func = OriginalImplementationProxyAnchor(num_classes, embedding_size, mrg = margin, alpha = alpha).to(device)

        loss_func.proxies = original_loss_func.proxies

        embedding_angles = list(range(0, 180))
        embeddings = torch.FloatTensor([c_f.angle_to_coord(a) for a in embedding_angles]).to(device) #2D embeddings
        labels = torch.randint(low=0, high=5, size=(180,)).to(device)

        loss = loss_func(embeddings, labels)
        correct_loss = original_loss_func(embeddings, labels)
        self.assertTrue(torch.isclose(loss, correct_loss))