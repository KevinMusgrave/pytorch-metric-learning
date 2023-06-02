import torch

from ..distances import DotProductSimilarity
from ..utils import common_functions as c_f
from .base_metric_loss_function import BaseMetricLossFunction
from torch_scatter import scatter_add


class ManifoldProxyLoss(BaseMetricLossFunction):
    def __init__(self, p: torch.tensor, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_not_supported(labels, ref_labels)
        ref_emb, indices_tuple, self.p = c_f.to_device((ref_emb, indices_tuple, self.p), tensor=embeddings, dtype=torch.float32)

        K = ref_emb.shape[0]
        proxies_by_embeddings = ref_emb[indices_tuple, :]
        p_by_embeddings = self.p[indices_tuple, :]
        loss = torch.exp(torch.sum(p_by_embeddings * embeddings, dim=1) -
                         torch.sum(p_by_embeddings * proxies_by_embeddings, dim=1))
        loss = scatter_add(loss, indices_tuple, dim_size=K).view(K, 1)
        loss = torch.log(1 + loss)

        return {
            "loss": {
                "losses": loss,
                "indices": c_f.torch_arange_from_size(ref_emb),
                "reduction_type": "element",
            }
        }

    def get_default_distance(self):
        return DotProductSimilarity()


class ManifoldLoss(BaseMetricLossFunction):
    r"""
    The parameters are defined as in the paper https://openaccess.thecvf.com/content_CVPR_2019/papers/Aziere_Ensemble_Deep_Manifold_Similarity_Learning_Using_Hard_Proxies_CVPR_2019_paper.pdf
    K: number of proxies.

    - l: embedding size.

    - method: type of manifold loss to compute. Available choices are ['intrinsic', 'context'].

    - alpha: parameter of the Random Walk. It is contained in :math:`(0,1)` and specifies the amount of similarity from
           each node is passed to neighbor nodes.

    - margin: margin used in the calculation of the loss.
    """

    def __init__(self, K=50, l=128, method="intrinsic", alpha=0.8, margin=5e-4, **kwargs):
        super().__init__(**kwargs)
        if 'intrinsic' not in method and 'context' not in method:
            raise TypeError(f"Method {method} not supported. Available choices are ['intrinsic', 'context'].")

        self.K = K
        self.l = l
        self.proxies = torch.zeros(K, l, requires_grad=True)
        self.method = method
        self.alpha = alpha
        self.margin = margin

        self.p = torch.randn(K, l, requires_grad=True)
        self.proxies_loss = self.get_default_proxy_loss(self.p)
        self.proxies_optimizer = self.get_default_optimizer(self.p)
        self.add_to_recordable_attributes(list_of_names=["K", "l", "method", "alpha", "margin"], is_stat=True)

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        c_f.labels_not_supported(labels, ref_labels)

        assert embeddings.shape[1] == self.l

        N = len(embeddings)
        if indices_tuple is not None:
            meta_classes = indices_tuple
        else:
            meta_classes = torch.randint(0, self.K, (N - self.K,))
            meta_classes = torch.cat((torch.arange(self.K), meta_classes))
            meta_classes = meta_classes[torch.randperm(N)]

        if torch.all(self.proxies == 0):
            self.proxies = self.get_default_proxies(embeddings, meta_classes)

        self.proxies = c_f.to_device(self.proxies, tensor=embeddings)
        embs_and_proxies = torch.cat([embeddings, self.proxies], dim=0)
        S = self.distance(embs_and_proxies, embs_and_proxies)
        D_inv_half = torch.pow(torch.abs(torch.sum(S, dim=1)),
                               -1 / 2)  # In the paper it is not specified how to avoid averall negative scalar products
        S_bar = D_inv_half * S.t()
        S_bar = S_bar.t() * D_inv_half
        S_bar.fill_diagonal_(0)
        F = (1 - self.alpha) * torch.inverse(torch.eye(N + self.K, N + self.K) - self.alpha * S_bar)[:, -self.K:]

        if 'intrinsic' in self.method:
            F = F[:N, :]
            loss = torch.exp(F - F[torch.arange(N), meta_classes].view(N, 1) + self.margin)
            loss = torch.log(1 + torch.sum(loss, dim=1))
            loss = loss.mean()
        else:
            proxies_repeted = F[-self.K:, :].unsqueeze(0).repeat(N, 1, 1)
            loss = torch.exp(ManifoldLoss.s(F[:N, :].unsqueeze(-1), proxies_repeted) -
                             ManifoldLoss.s(F[:N, :], F[N + meta_classes, :]).unsqueeze(-1) + self.margin)
            loss = torch.log(1 + torch.sum(loss, dim=1))
            loss = loss.mean()

        old_proxies = self.proxies.detach()
        old_embs = embeddings.detach()
        for _ in range(20):
            proxies_loss = self.proxies_loss(old_embs, None, meta_classes, old_proxies, None)
            self.proxies_optimizer.zero_grad()
            proxies_loss.backward()
            self.proxies_optimizer.step()
        self.proxies = self.p

        return {
            "loss": {
                "losses": loss,
                "indices": None,
                "reduction_type": "already_reduced",
            }
        }

    @staticmethod
    def s(x, p):
        return torch.sum(x * p, dim=1)

    def get_default_distance(self):
        return DotProductSimilarity()

    def get_default_proxies(self, embs, meta_classes):
        self.proxies = []
        for k in range(self.K):
            meta_class_k = embs[meta_classes == k, :]
            meta_class_k_row_index = torch.randint(meta_class_k.shape[0], (1,))
            self.proxies.append(meta_class_k[meta_class_k_row_index, :])
        return torch.cat(self.proxies, dim=0)

    def get_default_optimizer(self, p):
        return torch.optim.SGD([p], lr=0.01, momentum=0.9)

    def get_default_proxy_loss(self, p):
        return ManifoldProxyLoss(p)


