import torch

from .base_loss_wrapper import BaseLossWrapper


class SelfSupervisedLoss(BaseLossWrapper):
    def __init__(self, loss, symmetric=True, **kwargs):
        super().__init__(loss=loss, **kwargs)
        self.loss = loss
        self.symmetric = symmetric

    @staticmethod
    def supported_losses():
        return [
            "AngularLoss",
            "CircleLoss",
            "ContrastiveLoss",
            "GeneralizedLiftedStructureLoss",
            "IntraPairVarianceLoss",
            "MultiSimilarityLoss",
            "NTXentLoss",
            "SignalToNoiseRatioContrastiveLoss",
            "SupConLoss",
            "TripletMarginLoss",
            "NCALoss",
            "TupletMarginLoss",
        ]

    @classmethod
    def check_loss_support(cls, loss_name):
        if loss_name not in cls.supported_losses():
            raise ValueError(f"SelfSupervisedLoss not supported for {loss_name}")

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
