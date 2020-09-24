######################################
#######ORIGINAL IMPLEMENTATION########
######################################
# FROM https://github.com/idstcv/SoftTriple/blob/master/loss/SoftTriple.py
# This code is copied directly from the official implementation
# so that we can make sure our implementation returns the same result.
# It's copied under the Apache license.
# No changes have been made, except removal of .cuda() calls
# Implementation of SoftTriple Loss
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
from pytorch_metric_learning.utils import common_functions as c_f


class OriginalImplementationSoftTriple(nn.Module):
    def __init__(self, la, gamma, tau, margin, dim, cN, K):
        super(OriginalImplementationSoftTriple, self).__init__()
        self.la = la
        self.gamma = 1.0 / gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = Parameter(torch.Tensor(dim, cN * K))
        self.weight = torch.zeros(cN * K, cN * K, dtype=torch.bool)
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i * K + j, i * K + j + 1 : (i + 1) * K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc * self.gamma, dim=2)
        simClass = torch.sum(prob * simStruc, dim=2)
        marginM = torch.zeros(simClass.shape, dtype=input.dtype).to(input.device)
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la * (simClass - marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            small_val = c_f.small_val(input.dtype)
            simCenterMasked = torch.clamp(2.0 * simCenter[self.weight], max=2)
            reg = torch.sum(torch.sqrt(2.0 + small_val - simCenterMasked)) / (
                self.cN * self.K * (self.K - 1.0)
            )
            return lossClassify + self.tau * reg
        else:
            return lossClassify


import unittest
from .. import TEST_DTYPES
from pytorch_metric_learning.losses import SoftTripleLoss
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.regularizers import SparseCentersRegularizer


class TestSoftTripleLoss(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.device = torch.device("cuda")

    def test_soft_triple_loss(self):
        embedding_size = 2
        num_classes = 11
        reg_weight = 0.2
        margin = 0.01

        for dtype in TEST_DTYPES:
            la = 1 if dtype == torch.float16 else 20
            gamma = 1 if dtype == torch.float16 else 0.1
            for centers_per_class in range(1, 12):
                if centers_per_class > 1:
                    weight_regularizer = SparseCentersRegularizer(
                        num_classes, centers_per_class
                    )
                else:
                    weight_regularizer = None
                loss_func = SoftTripleLoss(
                    num_classes,
                    embedding_size,
                    centers_per_class=centers_per_class,
                    la=la,
                    gamma=gamma,
                    margin=margin,
                    weight_regularizer=weight_regularizer,
                    weight_reg_weight=reg_weight,
                ).to(self.device)
                original_loss_func = OriginalImplementationSoftTriple(
                    la,
                    gamma,
                    reg_weight,
                    margin,
                    embedding_size,
                    num_classes,
                    centers_per_class,
                ).to(self.device)

                original_loss_func.fc.data = original_loss_func.fc.data.type(dtype)
                loss_func.fc = original_loss_func.fc

                embedding_angles = torch.arange(0, 180)
                embeddings = torch.tensor(
                    [c_f.angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(
                    self.device
                )  # 2D embeddings
                labels = torch.randint(low=0, high=10, size=(180,)).to(self.device)

                loss = loss_func(embeddings, labels)
                loss.backward()
                correct_loss = original_loss_func(embeddings, labels)
                self.assertTrue(torch.isclose(loss, correct_loss))
