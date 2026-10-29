import unittest

import torch
from torch import nn

from pytorch_metric_learning.losses import P2SGradLoss

from .. import TEST_DEVICE, TEST_DTYPES, helpers


class TrustedImplementationP2SActivationLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        weight = torch.empty(in_dim, out_dim).uniform_(-1, 1)
        self.weight = nn.Parameter(weight.renorm_(2, 1, 1e-5).mul_(1e5))

    def forward(self, input_feat):
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        w = w.to(device=input_feat.device, dtype=input_feat.dtype)

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
            index = index.to(device=input_score.device, dtype=torch.long)
            target = target.to(device=input_score.device, dtype=torch.long)
            index.scatter_(1, target.view(-1, 1), 1)

        index = index.to(input_score.dtype)
        loss = self.m_loss(input_score, index)

        return loss


class TestP2SGradLoss(unittest.TestCase):
    def test_accumulated_forwards_preserve_parameter_storage_and_gradients(self):
        torch.manual_seed(31)
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                loss_func = P2SGradLoss(4, 3).to(TEST_DEVICE, dtype)
                parameter = loss_func.weight
                storage = parameter.data_ptr()
                inputs = [
                    torch.randn(
                        5, 3, device=TEST_DEVICE, dtype=dtype, requires_grad=True
                    )
                    for _ in range(2)
                ]
                labels = torch.tensor([0, 1, 2, 3, 0], device=TEST_DEVICE)
                actual, expected, snapshots = [], [], []
                for embeddings in inputs:
                    actual.append(loss_func(embeddings, labels))
                    self.assertIs(loss_func.weight, parameter)
                    self.assertEqual(parameter.data_ptr(), storage)
                    weights = parameter.detach().clone().requires_grad_()
                    snapshots.append(weights)
                    scores = (
                        torch.nn.functional.normalize(embeddings, dim=1) @ weights.t()
                    ).clamp(-1, 1)
                    targets = torch.nn.functional.one_hot(labels, 4).to(dtype=dtype)
                    expected.append((scores - targets).square().mean())
                actual_gradients = torch.autograd.grad(
                    sum(actual), [parameter, *inputs]
                )
                expected_gradients = torch.autograd.grad(
                    sum(expected), [*snapshots, *inputs]
                )
                tolerance = (
                    {"rtol": 1e-2, "atol": 1e-3} if dtype == torch.float16 else {}
                )
                torch.testing.assert_close(sum(actual), sum(expected), **tolerance)
                torch.testing.assert_close(
                    actual_gradients[0],
                    expected_gradients[0] + expected_gradients[1],
                    **tolerance,
                )
                for actual_gradient, expected_gradient in zip(
                    actual_gradients[1:], expected_gradients[2:], strict=True
                ):
                    torch.testing.assert_close(
                        actual_gradient, expected_gradient, **tolerance
                    )

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
                num_classes=num_classes, embedding_size=descriptors_dim
            ).to(TEST_DEVICE, dtype)
            copy_weights = loss_fn.weight.detach().clone()

            loss = loss_fn(embeddings, labels)
            (weight_grad,) = torch.autograd.grad(loss, loss_fn.weight)

            copy_weights = copy_weights.to(device=TEST_DEVICE, dtype=dtype)
            for j in range(num_classes):
                w_j = copy_weights[j]
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
                    torch.all(torch.isclose(gradients, weight_grad[j], rtol=0.5e-1))
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

            loss_func = P2SGradLoss(num_classes, descriptors_dim).to(TEST_DEVICE, dtype)
            t_layer = TrustedImplementationP2SActivationLayer(
                descriptors_dim, num_classes
            ).to(TEST_DEVICE, dtype)
            t_loss_func = TrustedImplementationP2SGradLoss()
            loss_func.weight = torch.nn.Parameter(
                t_layer.weight.detach().t().contiguous()
            )

            rtol = helpers.rtol(dtype)

            loss = loss_func(embeddings, labels)
            loss.backward()
            t_loss = t_loss_func(t_layer(embeddings), labels)
            t_loss.backward()
            self.assertTrue(torch.isclose(loss, t_loss, rtol=rtol))
