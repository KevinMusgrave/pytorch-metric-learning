from collections import defaultdict

import torch
from torch import nn

from ..distances import CosineSimilarity
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


class MultipleNegativesRankingLoss(BaseMetricLossFunction):
    """
    Args:
        scale: The scale factor for the loss
        positives_per_anchor: The number of positives per element to sample within a
            batch. Can be an integer or the string "all".

    Reference:
        `Henderson et al. 2017 <https://arxiv.org/pdf/1705.00652.pdf>`
    """

    def __init__(self, scale: float = 20.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def compute_loss(
        self,
        embeddings,
        labels,
        indices_tuple,
        ref_emb,
        ref_labels,
    ):
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)

        if indices_tuple is None:
            a, pos, a_neg_idx, neg = lmu.get_all_pairs_indices(labels, labels)
        else:
            a, pos, a_neg_idx, neg = lmu.convert_to_pairs(indices_tuple, labels, labels)

        anchor_losses = defaultdict(list)
        for idx, anc in enumerate(a):
            pos_idx = pos[idx]
            neg_idx = neg[a_neg_idx == anc]
            anchor_losses[anc.item()].append(
                self.calculate_anchor_positive_loss(embeddings, anc, pos_idx, neg_idx)
            )

        loss = torch.stack([(torch.stack(v).mean()) for v in anchor_losses.values()])
        return {
            "loss": {
                "losses": loss,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }

    def calculate_anchor_positive_loss(
        self, embeddings, anchor_idx, positive_idx, negative_idx
    ):
        """
        Calculates the loss for a given anchor-positive pair and negative samples.

        Args:
            embeddings: tensor of shape (batch_size, embedding_size)
            anchor_idx: index of the anchor embedding in the embeddings tensor
            positive_idx: index of the positive embedding in the embeddings tensor
            negative_idx: index of the negative embeddings in the embeddings tensor

        Returns:
            loss: tensor representing the calculated loss for the given anchor-positive pair and negative samples
        """
        logits = self.get_logits(embeddings, anchor_idx, positive_idx, negative_idx)
        target = torch.zeros(len(logits)).to(embeddings.device)
        target[0] = 1
        return self.cross_entropy_loss(logits, target)

    def get_logits(self, embeddings, anchor_idx, positive_idx, negative_idx):
        """
        Calculates the pairwise similarities between embeddings using the specified
        distance function. Those similarities will be treated as logits in the cross
        entropy loss that this class uses.

        Args:
            embeddings: tensor of shape (batch_size, embedding_size)

        Returns:
            pairwise_sims: tensor of shape (batch_size, batch_size) containing the
                pairwise similarities between embeddings
        """
        anchor_embedding = embeddings[anchor_idx].reshape(1, -1)
        positive_embedding = embeddings[positive_idx].reshape(1, -1)
        negative_embeddings = embeddings[negative_idx]
        positive_pairwise_sims = self.get_positive_pairwise_sims(
            anchor_embedding, positive_embedding
        )
        negative_pairwise_sims = self.get_positive_pairwise_sims(
            anchor_embedding, negative_embeddings
        )
        return torch.cat(
            [positive_pairwise_sims.flatten(), negative_pairwise_sims.flatten()]
        )

    def get_positive_pairwise_sims(self, anchor_embedding, ref_embedding):
        const = 1.0 if self.distance.is_inverted else -1.0
        return self.distance(anchor_embedding, ref_embedding) * const

    def get_default_distance(self):
        return CosineSimilarity()
