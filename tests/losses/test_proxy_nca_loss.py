import unittest

import torch
from torch.nn import functional as F

from pytorch_metric_learning.losses import ProxyNCALoss

from .. import TEST_DEVICE, TEST_DTYPES, helpers
from ..helpers import angle_to_coord


class TestProxyNCALoss(unittest.TestCase):
    def test_autocast_preserves_parameters_and_matches_reference_gradients(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(234)
                loss_func = ProxyNCALoss(4, 8).to(TEST_DEVICE)
                encoder = torch.nn.Linear(8, 8).to(TEST_DEVICE)
                parameters = (*encoder.parameters(), loss_func.weight)
                optimizer = torch.optim.Adam(parameters, lr=0.01)
                labels = torch.arange(4, device=TEST_DEVICE).repeat_interleave(3)
                inputs = torch.randn(12, 8, device=TEST_DEVICE)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(TEST_DEVICE.type, dtype=dtype):
                        embeddings = encoder(inputs)
                        actual = loss_func(embeddings, labels)
                        logits = loss_func.get_logits(embeddings)
                    self.assertEqual(embeddings.dtype, dtype)
                    normalized_embeddings = F.normalize(embeddings.float(), dim=1)
                    normalized_proxies = F.normalize(loss_func.weight, dim=1)
                    distances = (
                        (normalized_embeddings[:, None] - normalized_proxies[None])
                        .square()
                        .sum(dim=2)
                    )
                    expected = F.cross_entropy(-distances, labels)
                    torch.testing.assert_close(logits, distances)
                    torch.testing.assert_close(actual, expected)
                    expected_grads = torch.autograd.grad(
                        expected, parameters, retain_graph=True
                    )
                    actual.backward()
                    for parameter, expected_grad in zip(
                        parameters, expected_grads, strict=True
                    ):
                        torch.testing.assert_close(parameter.grad, expected_grad)
                    optimizer.step()
                    self.assertIs(loss_func.weight, parameters[-1])
                    self.assertEqual(loss_func.weight.dtype, torch.float32)
                    self.assertEqual(
                        optimizer.state[loss_func.weight]["exp_avg"].dtype,
                        torch.float32,
                    )

    def test_proxy_nca_loss(self):
        torch.manual_seed(14892)
        for dtype in TEST_DTYPES:
            softmax_scale = 1 if dtype == torch.float16 else 10
            loss_func = ProxyNCALoss(
                softmax_scale=softmax_scale, num_classes=10, embedding_size=2
            ).to(TEST_DEVICE, dtype)

            embedding_angles = torch.arange(0, 180)
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(TEST_DEVICE)
            labels = torch.randint(low=0, high=10, size=(180,))

            loss = loss_func(embeddings, labels)
            loss.backward()

            normalized_embeddings = torch.nn.functional.normalize(embeddings)
            proxies = torch.nn.functional.normalize(loss_func.weight, p=2, dim=1)
            correct_loss = torch.zeros((), dtype=dtype, device=TEST_DEVICE)
            for i in range(len(normalized_embeddings)):
                curr_emb, curr_label = normalized_embeddings[i], labels[i]
                curr_proxy = proxies[curr_label]
                denominator = torch.sum((curr_emb - proxies) ** 2, dim=1)
                denominator = torch.sum(torch.exp(-denominator * softmax_scale))
                numerator = torch.sum((curr_emb - curr_proxy) ** 2)
                numerator = torch.exp(-numerator * softmax_scale)
                correct_loss += -torch.log(numerator / denominator)

            correct_loss /= len(embeddings)
            rtol = helpers.rtol(dtype)
            self.assertTrue(torch.isclose(loss, correct_loss, rtol=rtol))

            if dtype != torch.float16:
                logits_out = loss_func.get_logits(embeddings)
                self.assertTrue(
                    torch.allclose(
                        logits_out,
                        torch.cdist(normalized_embeddings, proxies) ** 2,
                        rtol=1e-2,
                    )
                )
