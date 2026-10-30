import unittest
from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from pytorch_metric_learning.losses import ProxyAnchorLoss

from .. import TEST_DEVICE, TEST_DTYPES, helpers
from ..helpers import angle_to_coord


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing

    T = sklearn.preprocessing.label_binarize(T, classes=range(nb_classes))
    T = torch.as_tensor(T, dtype=torch.float32)
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
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed))
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes).to(X.device)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.where(P_one_hot.sum(dim=0) != 0)[0]
        num_valid_proxies = len(with_pos_proxies)

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(
            dim=0
        )
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(
            dim=0
        )

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss


class TestProxyAnchorLoss(unittest.TestCase):
    def test_proxyanchor_loss(self):
        self.check_proxyanchor_loss(use_autocast=False)

    @pytest.mark.cuda
    def test_proxyanchor_loss_under_cuda_autocast(self):
        self.check_proxyanchor_loss(use_autocast=True)

    def check_proxyanchor_loss(self, use_autocast):
        num_classes = 10
        embedding_size = 2
        margin = 0.5
        alpha = 32
        cm = torch.amp.autocast("cuda") if use_autocast else nullcontext()

        for dtype in TEST_DTYPES:
            loss_func = ProxyAnchorLoss(
                num_classes, embedding_size, margin=margin, alpha=alpha
            ).to(TEST_DEVICE)
            original_loss_func = OriginalImplementationProxyAnchor(
                num_classes, embedding_size, mrg=margin, alpha=alpha
            ).to(TEST_DEVICE, torch.float64)

            if not use_autocast:
                loss_func.to(dtype=dtype)
            with torch.no_grad():
                loss_func.weight.copy_(original_loss_func.proxies)
                original_loss_func.proxies.copy_(loss_func.weight)

            embedding_angles = list(range(180))
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=torch.float32,
            ).to(TEST_DEVICE)

            if not use_autocast:
                embeddings = embeddings.type(dtype)
            labels = torch.randint(low=0, high=5, size=(180,)).to(TEST_DEVICE)

            with cm:
                loss = loss_func(embeddings, labels)

            loss.backward()
            exact_embeddings = embeddings.detach().double()
            correct_loss = original_loss_func(exact_embeddings, labels)
            helpers.assert_close(
                loss, correct_loss, torch.float16 if use_autocast else dtype, atol=1e-8
            )

            logits_out = loss_func.get_logits(embeddings)
            cosines = torch.matmul(
                F.normalize(exact_embeddings),
                F.normalize(original_loss_func.proxies).t(),
            )
            helpers.assert_close(logits_out, cosines, logits_out.dtype)


def weighted_proxy_anchor_reference(embeddings, labels, proxies, weights):
    margin, alpha = 0.1, 32
    cos = F.normalize(embeddings, dim=1) @ F.normalize(proxies, dim=1).T
    positive = F.one_hot(labels, len(proxies)).bool()
    used = weights[:, None] > 0
    log_w = torch.log(weights)[:, None]
    pos = (-alpha * (cos - margin) + log_w).masked_fill(~(positive & used), -torch.inf)
    neg = (alpha * (cos + margin) + log_w).masked_fill(~(~positive & used), -torch.inf)
    pos_term = F.softplus(torch.logsumexp(pos, 0))
    neg_term = F.softplus(torch.logsumexp(neg, 0))
    with_positives = positive.any(0)
    return pos_term[with_positives].sum() / with_positives.sum() + neg_term.mean()


class TestProxyAnchorMinerWeights(unittest.TestCase):
    def test_mined_weights_scale_exponential_terms(self):
        labels = torch.tensor([0, 0, 1, 2, 2, 1], device=TEST_DEVICE)
        anchors = torch.tensor([0, 0, 2, 3], device=TEST_DEVICE)
        positives = torch.tensor([1, 1, 5, 4], device=TEST_DEVICE)
        negatives = torch.tensor([2, 3, 0, 1], device=TEST_DEVICE)
        triplets = (anchors, positives, negatives)
        counts = torch.bincount(torch.cat(triplets), minlength=len(labels))
        weights = counts.double() / counts.max()
        embeddings = torch.randn(6, 4)
        for dtype in TEST_DTYPES:
            loss_func = ProxyAnchorLoss(3, 4).to(TEST_DEVICE, dtype)
            proxies = loss_func.weight.detach().double()
            helpers.assert_matches_float64(
                lambda x, loss_func=loss_func: loss_func(x, labels, triplets),
                lambda x, proxies=proxies: weighted_proxy_anchor_reference(
                    x, labels, proxies, weights.to(TEST_DEVICE)
                ),
                [embeddings],
                dtype,
                atol=1e-5,
            )


if __name__ == "__main__":
    unittest.main()
