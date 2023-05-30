import torch

from ..distances import DotProductSimilarity
from ..distances.manifold_similarity import ManifoldSimilarity
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


class ManifoldLoss(BaseMetricLossFunction):
    def __init__(self, method="intrinsic", alpha=0.8, margin=5e-4, **kwargs):
        super().__init__(**kwargs)
        # TODO: AGGIUNGERE IL CONTROLLO CHE SIA INTRINSIC O CONTEXT
        self.method = method
        self.alpha = alpha
        self.margin = margin
        self.add_to_recordable_attributes(name="num_pairs", is_stat=True)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

    # https://openaccess.thecvf.com/content_CVPR_2019/papers/Aziere_Ensemble_Deep_Manifold_Similarity_Learning_Using_Hard_Proxies_CVPR_2019_paper.pdf
    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        # TODO: AGGIUNGI MODULO PER MIGLIORARE LE PROXIES E RIMUOVERE LA DIPENDENZA DA REF_EMB E REF_LABELS
        # c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        N, K = len(embeddings), len(ref_emb)
        S = self.distance(embeddings, ref_emb)
        D_inv_half = torch.pow(torch.sum(S, dim=2), -1/2)
        S_bar = D_inv_half * S.t()
        S_bar = S_bar.t() * D_inv_half
        F = (1-self.alpha)*torch.inverse(torch.eye(N+K, N+K) - self.alpha*S_bar)[:, N+1:]
        loss = (torch.log(1 + torch.sum(torch.exp(F - F[:, ref_labels] + self.margin)))).mean()
        return {
            "loss": {
                "losses": loss,
                "indices": None,
                "reduction_type": "already_reduced",
            }
        }

    def get_default_distance(self):
        return ManifoldSimilarity()
