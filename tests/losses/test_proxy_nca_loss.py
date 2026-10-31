import unittest

import torch
from torch.nn import functional as F

from pytorch_metric_learning.losses import ProxyNCALoss
from pytorch_metric_learning.types import TripletIndices

from .. import TEST_DEVICE, TEST_DTYPES, helpers
from ..helpers import angle_to_coord


def proxy_nca_plus_plus_equation_6(embeddings, proxies, labels, temperature):
    x = embeddings / embeddings.norm(dim=1, keepdim=True)
    p = proxies / proxies.norm(dim=1, keepdim=True)
    terms = []
    for row, label in enumerate(labels.tolist()):
        scaled = [
            torch.exp(-(x[row] - p[z]).pow(2).sum() / temperature)
            for z in range(len(p))
        ]
        terms.append(-torch.log(scaled[label] / torch.stack(scaled).sum()))
    return torch.stack(terms).mean()


class TestProxyNCALoss(unittest.TestCase):
    def test_softmax_scale_is_the_proxy_nca_plus_plus_objective(self):
        labels = torch.tensor([0, 1, 2, 3, 1, 0], device=TEST_DEVICE)
        generator = torch.Generator().manual_seed(2)
        rows = torch.randn(6, 5, generator=generator, dtype=torch.float64)
        proxies = torch.randn(4, 5, generator=generator, dtype=torch.float64)
        for dtype in TEST_DTYPES:
            with self.subTest(dtype=dtype):
                loss_func = ProxyNCALoss(4, 5, softmax_scale=9).to(TEST_DEVICE, dtype)
                with torch.no_grad():
                    loss_func.weight.copy_(proxies.to(TEST_DEVICE, dtype))

                actual = loss_func(rows.to(TEST_DEVICE, dtype), labels)
                expected = proxy_nca_plus_plus_equation_6(
                    rows.to(TEST_DEVICE, dtype).double(),
                    proxies.to(TEST_DEVICE, dtype).double(),
                    labels,
                    1 / 9,
                )
                helpers.assert_close(actual.detach(), expected, dtype)
                embeddings = rows.to(TEST_DEVICE, dtype).requires_grad_()
                wide = embeddings.detach().double().requires_grad_()
                wide_proxies = loss_func.weight.detach().double().requires_grad_()
                loss_func.zero_grad()
                loss_func(embeddings, labels).backward()
                wide_gradients = torch.autograd.grad(
                    proxy_nca_plus_plus_equation_6(wide, wide_proxies, labels, 1 / 9),
                    (wide, wide_proxies),
                )
                assert embeddings.grad is not None
                assert loss_func.weight.grad is not None
                helpers.assert_close(embeddings.grad, wide_gradients[0], dtype)
                helpers.assert_close(loss_func.weight.grad, wide_gradients[1], dtype)

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
            labels = torch.randint(low=0, high=10, size=(180,)).to(TEST_DEVICE)

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


class TestProxyNCAMinerWeights(unittest.TestCase):
    def test_every_tuple_position_counts_toward_the_weight(self):
        torch.manual_seed(5)
        loss_func = ProxyNCALoss(3, 4).to(TEST_DEVICE, torch.float64)
        embeddings = torch.randn(6, 4, device=TEST_DEVICE, dtype=torch.float64)
        labels = torch.tensor([0, 0, 1, 1, 2, 2], device=TEST_DEVICE)
        triplets = TripletIndices(
            torch.tensor([0, 0, 2], device=TEST_DEVICE),
            torch.tensor([1, 1, 3], device=TEST_DEVICE),
            torch.tensor([2, 4, 4], device=TEST_DEVICE),
        )
        loss = loss_func(embeddings, labels, triplets)
        proxies = F.normalize(loss_func.weight, dim=1)
        distances = torch.cdist(F.normalize(embeddings, dim=1), proxies) ** 2
        per_row = F.cross_entropy(-distances, labels, reduction="none")
        counts = torch.tensor([2, 2, 2, 1, 2, 0], device=TEST_DEVICE)
        expected = (per_row * counts / 2).mean()
        torch.testing.assert_close(loss, expected.to(loss.dtype))
