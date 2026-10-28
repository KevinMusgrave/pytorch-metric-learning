import torch

from .angular_loss import AngularLoss
from .circle_loss import CircleLoss
from .contrastive_loss import ContrastiveLoss
from .intra_pair_variance_loss import IntraPairVarianceLoss
from .lifted_structure_loss import GeneralizedLiftedStructureLoss
from .multi_similarity_loss import MultiSimilarityLoss
from .nca_loss import NCALoss
from .ntxent_loss import NTXentLoss
from .proxy_nca_loss import ProxyNCALoss
from .signal_to_noise_ratio_losses import SignalToNoiseRatioContrastiveLoss
from .supcon_loss import SupConLoss
from .triplet_margin_loss import TripletMarginLoss
from .tuplet_margin_loss import TupletMarginLoss


class SelfSupervisedLoss(torch.nn.Module):
    def __init__(self, loss, symmetric=True, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(loss, self.supported_losses()) or isinstance(
            loss, ProxyNCALoss
        ):
            raise TypeError(
                f"SelfSupervisedLoss not supported for {type(loss).__name__}"
            )
        self.loss = loss
        self.symmetric = symmetric

    @staticmethod
    def supported_losses():
        return (
            AngularLoss,
            CircleLoss,
            ContrastiveLoss,
            GeneralizedLiftedStructureLoss,
            IntraPairVarianceLoss,
            MultiSimilarityLoss,
            NTXentLoss,
            SignalToNoiseRatioContrastiveLoss,
            SupConLoss,
            TripletMarginLoss,
            NCALoss,
            TupletMarginLoss,
        )

    def forward(self, embeddings, ref_emb):
        labels = torch.arange(embeddings.shape[0])
        if self.symmetric:
            embeddings = torch.cat([embeddings, ref_emb], dim=0)
            labels = torch.cat([labels, labels], dim=0)
            return self.loss(embeddings, labels)
        return self.loss(
            embeddings=embeddings,
            labels=labels,
            ref_emb=ref_emb,
            ref_labels=labels.clone(),
        )
