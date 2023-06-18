import torch

from .base_loss_wrapper import BaseLossWrapper


class SelfSupervisedLoss(BaseLossWrapper):
    """
    Issue #411:

    A common use case is to have embeddings and ref_emb be augmented versions of each other.
    For most losses right now you have to create labels to indicate
    which embeddings correspond with which ref_emb.
    A wrapper that does this for the user would be nice.

        loss_fn = SelfSupervisedLoss(TripletMarginLoss())
        loss = loss_fn(embeddings, ref_emb1, ref_emb2, ...)

    where ref_embk = kth augmentation of embeddings.
    """

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
            "LiftedStructureLoss",
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
            raise Exception(f"SelfSupervisedLoss not supported for {loss_name}")

    def forward(self, embeddings, ref_emb):
        """
        embeddings: representations of the original set of inputs
        ref_emb:    representations of an augmentation of the inputs.
        *args:      variable length argument list, where each argument
                    is an additional representation of an augmented version of the input.
                    i.e. ref_emb2, ref_emb3, ...
        """
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
