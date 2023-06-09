# /usr/bin/env python3
import numpy as np
import torch
from torch import nn

from ..distances import CosineSimilarity
from ..utils import common_functions as c_f
from .base_metric_loss_function import BaseMetricLossFunction


class ManifoldLoss(BaseMetricLossFunction):
    r"""
    The parameters are defined as in the paper https://openaccess.thecvf.com/content_CVPR_2019/papers/Aziere_Ensemble_Deep_Manifold_Similarity_Learning_Using_Hard_Proxies_CVPR_2019_paper.pdf
    - l: embedding size.

    - K: number of proxies. Optional

    - lambdaC: regularization weight. Used in the formula
                :math:`loss = loss^{int} + \lambda_C*loss^{ctx}`.
        For :math:`\lambda_C=0` use only intrinsic loss, for :math:`\lambda_C=\infty` use only context loss.
        Optional

    - alpha: parameter of the Random Walk. It is contained in :math:`(0,1)` and specifies the amount of similarity from
           each node is passed to neighbor nodes. Optional

    - margin: margin used in the calculation of the loss. Optional
    """

    def __init__(
        self,
        l: int,
        K: int = 50,
        lambdaC: float = 1.0,
        alpha: float = 0.8,
        margin: float = 5e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if lambdaC < 0:
            raise ValueError(
                f"Uncorrect value for lambdaC argument. "
                f"Given lambdaC={lambdaC} but accepted only non-negative values"
            )

        self.K = K
        self.l = l
        self.proxies = nn.Parameter(torch.randn(K, l))
        self.lambdaC = lambdaC
        self.alpha = alpha
        self.margin = margin
        self.add_to_recordable_attributes(
            list_of_names=["K", "l", "lambdaC", "alpha", "margin"], is_stat=True
        )

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

        loss_int = torch.zeros(1)
        loss_int = c_f.to_device(loss_int, tensor=embeddings, dtype=embeddings.dtype)
        embs_and_proxies = torch.cat([embeddings, self.proxies], dim=0)

        S = self.distance(embs_and_proxies, embs_and_proxies).clamp(0, np.inf)
        S = torch.exp(S / 0.5)
        Y = torch.eye(N + self.K, device=S.device, dtype=S.dtype)
        S = S - S * Y

        D_inv_half = torch.pow(torch.sum(S, dim=1), -1 / 2).diag()
        S_bar = D_inv_half.mm(S)
        S_bar = S_bar.mm(D_inv_half)

        dt = S_bar.dtype
        L = torch.inverse(
            Y.float() - self.alpha * S_bar.float()
        )  # Added float cast since inverse is not available for Half
        L = L.to(dt)

        F = (1 - self.alpha) * L

        F_p = F[N:, :].clone()
        F_e = F[:N, :].clone()
        if self.lambdaC != np.inf:
            F = F[:N, N:]
            loss_int = F - F[torch.arange(N), meta_classes].view(-1, 1) + self.margin
            loss_int[
                torch.arange(N), meta_classes
            ] = -np.inf  # This way avoid numerical cancellation happening   # NoQA
            # instead with subtraction of margin term           # NoQA
            loss_int[
                loss_int < 0
            ] = -np.inf  # This way no loss for positive correlation with own proxy

            loss_int = torch.exp(loss_int)
            loss_int = torch.log(1 + torch.sum(loss_int, dim=1))
            loss_int = loss_int.mean()

        loss_ctx = torch.nn.functional.cosine_similarity(
            F_e, F_p.unsqueeze(1), dim=-1
        ).t()
        loss_ctx += -loss_ctx[torch.arange(N), meta_classes].view(-1, 1) + self.margin
        loss_ctx[
            torch.arange(N), meta_classes
        ] = -np.inf  # This way avoid numerical cancellation happening   # NoQA
        # instead with subtraction of margin term           # NoQA
        loss_ctx[loss_ctx < 0] = -np.inf

        # This way no loss for positive correlation with own proxy
        loss_ctx = torch.exp(loss_ctx)
        loss_ctx = torch.log(1 + torch.sum(loss_ctx, dim=1))
        loss_ctx = loss_ctx.mean()
        loss = loss_int + self.lambdaC * loss_ctx

        return {
            "loss": {
                "losses": loss,
                "indices": None,
                "reduction_type": "already_reduced",
            }
        }

    def get_default_distance(self):
        return CosineSimilarity()
