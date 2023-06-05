#!/usr/bin/env python3
import unittest
import torch
import torch.nn as nn
from pytorch_metric_learning.losses import P2SGradLoss
from pytorch_metric_learning.utils import common_functions as c_f

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

from .. import TEST_DTYPES, TEST_DEVICE
from ..zzz_testing_utils.testing_utils import angle_to_coord


######################################
#######TRUSTED IMPLEMENTATION########
######################################
# FROM https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/blob/master/core_modules/p2sgrad.py
# This code is copied directly from the most trusted implementation
# on the web. This is the closest way to the one proposed in the paper.
# The paper specified that the method consisted in computing new values for the gradients
# but this technique is equivalent to an MSE loss over cosine angles coming from the network.
# The code has been adapted for tests.
# It's copied under the BSD license.
class TrustedImplementationP2SActivationLayer(nn.Module):
    """ Output layer that produces cos\theta between activation vector x
    and class vector w_j

    in_dim:     dimension of input feature vectors
    output_dim: dimension of output feature vectors
                (i.e., number of classes)



    Usage example:
      batch_size = 64
      input_dim = 10
      class_num = 5

      l_layer = P2SActivationLayer(input_dim, class_num)
      l_loss = P2SGradLoss()

      data = torch.rand(batch_size, input_dim, requires_grad=True)
      target = (torch.rand(batch_size) * class_num).clamp(0, class_num-1)
      target = target.to(torch.long)

      scores = l_layer(data)
      loss = l_loss(scores, target)

      loss.backward()
    """

    def __init__(self, in_dim, out_dim):
        super(TrustedImplementationP2SActivationLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True)
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        return

    def forward(self, input_feat):
        """
        Compute P2SGrad activation

        input:
        ------
          input_feat: tensor (batch_size, input_dim)

        output:
        -------
          tensor (batch_size, output_dim)

        """
        # normalize the weight (again)
        # w (feature_dim, output_dim)
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        w = c_f.to_device(w, tensor=input_feat, dtype=input_feat.dtype)

        # normalize the input feature vector
        # x_modulus (batch_size)
        # sum input -> x_modules in shape (batch_size)
        x_modulus = input_feat.pow(2).sum(1).pow(0.5)
        # w_modules (output_dim)
        # w_moduls should be 1, since w has been normalized
        # w_modulus = w.pow(2).sum(0).pow(0.5)

        # W * x = ||W|| * ||x|| * cos())))))))
        # inner_wx (batch_size, output_dim)
        inner_wx = input_feat.mm(w)
        # cos_theta (batch_size, output_dim)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)

        # done
        return cos_theta


######################################
#######TRUSTED IMPLEMENTATION########
######################################
# FROM https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/blob/master/core_modules/p2sgrad.py
# This code is copied directly from the most trusted implementation
# on the web. This is the closest way to the one proposed in the paper.
# The paper specified that the method consisted in computing new values for the gradients
# but this technique is equivalent to an MSE loss over cosine angles coming from the network.
# The code has been adapted for tests.
# It's copied under the BSD license.
class TrustedImplementationP2SGradLoss(nn.Module):
    """P2SGradLoss() MSE loss between output and target one-hot vectors

    See usage in __doc__ of P2SActivationLayer
    """

    def __init__(self):
        super(TrustedImplementationP2SGradLoss, self).__init__()
        self.m_loss = nn.MSELoss()

    def forward(self, input_score, target):
        r"""
        input
        -----
          input_score: tensor (batch_size, class_num)
                 cos \theta given by P2SActivationLayer(input_feat)
          target: tensor (batch_size)
                 target[i] is the target class index of the i-th sample

        output
        ------
          loss: scaler
        """

        # filling in the target
        # index (batch_size, class_num)
        with torch.no_grad():
            index = torch.zeros_like(input_score)
            index, target = c_f.to_device((index, target), tensor=input_score, dtype=torch.long)
            # index[i][target[i][j]] = 1
            index.scatter_(1, target.data.view(-1, 1), 1)

        # MSE between \cos\theta and one-hot vectors
        index = index.to(input_score.dtype)
        loss = self.m_loss(input_score, index)

        return loss


class TestP2SGradLoss(unittest.TestCase):

    def test_p2s_grad_loss_with_paper_formula(self):
        num_classes = 20
        batch_size = 100
        descriptors_dim = 2
        for dtype in TEST_DTYPES:
            embedding_angles = torch.arange(batch_size)
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = (torch.rand(batch_size) * num_classes).clamp(0, num_classes - 1)
            labels = labels.to(torch.long)

            loss_fn = P2SGradLoss(descriptors_dim=descriptors_dim, num_classes=num_classes)
            optimizer = torch.optim.SGD([loss_fn.weight], lr=0.001)
            optimizer.zero_grad()
            copy_weights = loss_fn.weight.data.clone()

            loss = loss_fn(embeddings, labels)
            loss.backward()

            copy_weights = c_f.to_device(copy_weights, device=TEST_DEVICE, dtype=dtype)
            for j in range(num_classes):
                w_j = copy_weights[:, j]
                x_norm = torch.norm(embeddings, p=2, dim=1).view(-1, 1)
                w_norm = torch.norm(w_j, p=2, dim=0)
                cos_theta_j = embeddings.mm(w_j.unsqueeze(1)) / (x_norm * w_norm)
                cos_theta_j = cos_theta_j.view(-1, 1)

                index = torch.zeros(labels.shape, device=TEST_DEVICE)
                index[labels == j] = 1
                D_j = embeddings / x_norm
                L_j = cos_theta_j - index.unsqueeze(1)

                gradients = 2 * torch.mean(L_j * D_j, dim=0) / num_classes
                gradients = gradients.to(dtype)
                self.assertTrue(torch.all(torch.isclose(gradients,
                                                        loss_fn.weight.grad[:, j], rtol=0.5e-1)))

    def test_p2s_grad_loss_with_trusted_implementation(self):

        for dtype in TEST_DTYPES:
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.LongTensor([0, 0, 1, 1, 2])
            class_num = len(labels.data.unique())
            descriptors_dim = 2
            input_dim = 2
            loss_func = P2SGradLoss(descriptors_dim, class_num)
            t_layer = TrustedImplementationP2SActivationLayer(input_dim, class_num)
            t_loss_func = TrustedImplementationP2SGradLoss()
            loss_func.weight = t_layer.weight  # Only to ensure they start from equal initializations

            rtol = 1e-2 if dtype == torch.float16 else 1e-5

            loss = loss_func(embeddings, labels)
            loss.backward()
            t_loss = t_loss_func(t_layer(embeddings), labels)
            t_loss.backward()
            self.assertTrue(torch.isclose(loss, t_loss, rtol=rtol))
