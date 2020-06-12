######################################
#######ORIGINAL IMPLEMENTATION########
######################################
# FROM https://github.com/idstcv/SoftTriple/blob/master/loss/SoftTriple.py
# This code is copied directly from the official implementation
# so that we can make sure our implementation returns the same result.
# It's copied under the Apache license.
# No changes have been made
# Implementation of SoftTriple Loss
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init


class OriginalImplementationSoftTriple(nn.Module):
    def __init__(self, la, gamma, tau, margin, dim, cN, K):
        super(OriginalImplementationSoftTriple, self).__init__()
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = Parameter(torch.Tensor(dim, cN*K))
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool)
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        marginM = torch.zeros(simClass.shape)
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify


import unittest
from pytorch_metric_learning.losses import SoftTripleLoss
from pytorch_metric_learning.utils import common_functions as c_f

class TestSoftTripleLoss(unittest.TestCase):
    def test_soft_triple_loss(self):
        embedding_size = 2
        num_classes = 11
        la = 20
        gamma = 0.1
        reg_weight = 0.2
        margin = 0.01

        for centers_per_class in range(1, 12):

            loss_func = SoftTripleLoss(embedding_size, num_classes, centers_per_class=centers_per_class, la=la, gamma=gamma, reg_weight=reg_weight, margin=margin)
            original_loss_func = OriginalImplementationSoftTriple(la, gamma, reg_weight, margin, embedding_size, num_classes, centers_per_class)

            loss_func.fc = original_loss_func.fc

            embedding_angles = torch.arange(0, 180, 1)
            embeddings = torch.FloatTensor([c_f.angle_to_coord(a) for a in embedding_angles]) #2D embeddings
            labels = torch.randint(low=0, high=10, size=(180,))

            loss = loss_func(embeddings, labels)
            correct_loss = original_loss_func(embeddings, labels)

            self.assertTrue(torch.isclose(loss, correct_loss))