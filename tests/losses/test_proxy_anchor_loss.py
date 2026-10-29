import unittest
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn

from pytorch_metric_learning.losses import ProxyAnchorLoss

from .. import TEST_DEVICE, TEST_DTYPES
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
        num_classes = 10
        embedding_size = 2
        margin = 0.5

        for use_autocast in [True, False]:
            cm = torch.amp.autocast("cuda") if use_autocast else nullcontext()

            for dtype in TEST_DTYPES:
                alpha = 1 if dtype == torch.float16 else 32
                loss_func = ProxyAnchorLoss(
                    num_classes, embedding_size, margin=margin, alpha=alpha
                ).to(TEST_DEVICE)
                original_loss_func = OriginalImplementationProxyAnchor(
                    num_classes, embedding_size, mrg=margin, alpha=alpha
                ).to(TEST_DEVICE)

                if not use_autocast:
                    original_loss_func.to(dtype=dtype)
                loss_func.weight = original_loss_func.proxies

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
                correct_loss = original_loss_func(embeddings, labels)
                rtol = 1e-2 if dtype == torch.float16 or use_autocast else 1e-5

                self.assertTrue(torch.isclose(loss, correct_loss, rtol=rtol))

                logits_out = loss_func.get_logits(embeddings)
                self.assertTrue(
                    torch.allclose(
                        logits_out,
                        torch.matmul(
                            F.normalize(embeddings), F.normalize(loss_func.weight).t()
                        ),
                    )
                )


if __name__ == "__main__":
    unittest.main()
