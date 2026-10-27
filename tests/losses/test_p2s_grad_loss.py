import unittest

import torch
from torch import nn

from pytorch_metric_learning.losses import P2SGradLoss
from pytorch_metric_learning.utils import common_functions as c_f

from .. import TEST_DEVICE, TEST_DTYPES


class TrustedImplementationP2SActivationLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True)
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, input_feat):
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        w = c_f.to_device(w, tensor=input_feat, dtype=input_feat.dtype)

        x_modulus = input_feat.pow(2).sum(1).pow(0.5)

        inner_wx = input_feat.mm(w)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)

        return cos_theta


class TrustedImplementationP2SGradLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_loss = nn.MSELoss()

    def forward(self, input_score, target):
        with torch.no_grad():
            index = torch.zeros_like(input_score)
            index = c_f.to_device(index, tensor=input_score, dtype=torch.long)
            target = c_f.to_device(target, tensor=input_score, dtype=torch.long)
            index.scatter_(1, target.data.view(-1, 1), 1)

        index = index.to(input_score.dtype)
        loss = self.m_loss(input_score, index)

        return loss


class TestP2SGradLoss(unittest.TestCase):
    def test_p2s_grad_loss_with_paper_formula(self):
        torch.manual_seed(23)
        num_classes = 20
        batch_size = 100
        descriptors_dim = 128
        for dtype in TEST_DTYPES:
            if dtype == torch.float16:
                continue
            embeddings = torch.randn(
                batch_size,
                descriptors_dim,
                device=TEST_DEVICE,
                dtype=dtype,
                requires_grad=True,
            )
            labels = torch.randint(
                0, num_classes, size=(batch_size,), device=TEST_DEVICE
            )

            loss_fn = P2SGradLoss(
                descriptors_dim=descriptors_dim, num_classes=num_classes
            ).to(TEST_DEVICE, dtype)
            copy_weights = loss_fn.weight.data.clone()

            loss = loss_fn(embeddings, labels)
            (weight_grad,) = torch.autograd.grad(loss, loss_fn.weight)

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
                self.assertTrue(
                    torch.all(torch.isclose(gradients, weight_grad[:, j], rtol=0.5e-1))
                )

    def test_p2s_grad_loss_with_trusted_implementation(self):
        torch.manual_seed(23)
        num_classes = 20
        batch_size = 100
        descriptors_dim = 128

        for dtype in TEST_DTYPES:
            embeddings = torch.randn(
                batch_size,
                descriptors_dim,
                device=TEST_DEVICE,
                dtype=dtype,
                requires_grad=True,
            )
            labels = torch.randint(
                0, num_classes, size=(batch_size,), device=TEST_DEVICE
            )

            loss_func = P2SGradLoss(descriptors_dim, num_classes).to(TEST_DEVICE, dtype)
            t_layer = TrustedImplementationP2SActivationLayer(
                descriptors_dim, num_classes
            ).to(TEST_DEVICE, dtype)
            t_loss_func = TrustedImplementationP2SGradLoss()
            loss_func.weight = t_layer.weight

            rtol = 1e-2 if dtype == torch.float16 else 1e-5

            loss = loss_func(embeddings, labels)
            loss.backward()
            t_loss = t_loss_func(t_layer(embeddings), labels)
            t_loss.backward()
            self.assertTrue(torch.isclose(loss, t_loss, rtol=rtol))
