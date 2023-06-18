import torch
import torch.nn as nn
from torch.nn import Parameter

from ..utils import common_functions as c_f
from . import BaseMetricLossFunction

# Adapted from https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/blob/master/core_modules/p2sgrad.py
# Original Author: Xin Wang
# Email: wangxin@nii.ac.jp
# Copyright: Copyright 2021, Xin Wang


class P2SGradLoss(BaseMetricLossFunction):
    r"""
    P2SGrad:
    https://arxiv.org/abs/1905.02479
    Zhang, X. et al. P2SGrad: Refined gradients for optimizing deep face models.
    in Proc. CVPR 9906-9914, 2019

    The gradient formulas defined in Eq.(11) of the paper are equivalent to use an MSE loss with 0
    or 1 as target:

    \mathcal{L}_i = \sum_{j=1}^{K} (\cos\theta_{i,j} - \delta(j == y_i))^2

    The difference from a common MSE is that the network output is cos angle.
    For this reason the gradient update becomes a new loss function tailored for face recognition systems.
    """

    def __init__(self, descriptors_dim, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.descriptors_dim = descriptors_dim
        self.num_classes = num_classes

        self.weight = Parameter(torch.Tensor(descriptors_dim, num_classes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.m_loss = nn.MSELoss()

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        c_f.indices_tuple_not_supported(indices_tuple)

        self.weight.data = self.weight.data.renorm(2, 1, 1e-5).mul(1e5)
        dtype = embeddings.dtype
        self.weight.data = c_f.to_device(
            self.weight.data, tensor=embeddings, dtype=dtype
        )

        rtol = 1e-2 if dtype == torch.float16 else 1e-5
        w_modulus = self.weight.pow(2).sum(0).pow(0.5)
        assert torch.all(torch.abs(w_modulus - 1.0) < rtol)

        # W * x = ||W|| * ||x|| * cos()
        x_modulus = embeddings.pow(2).sum(1).pow(0.5)
        cos_theta = embeddings.mm(self.weight)
        # cos_theta (batch_size, num_classes)
        cos_theta = cos_theta / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)

        with torch.no_grad():
            index = torch.zeros_like(cos_theta)
            index.scatter_(1, labels.view(-1, 1), 1)

        # MSE between \cos\theta and one-hot vectors
        loss = self.m_loss(cos_theta, index)

        return {
            "loss": {
                "losses": loss,
                "indices": None,
                "reduction_type": "already_reduced",
            }
        }
