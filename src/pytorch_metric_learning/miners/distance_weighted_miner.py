import torch

from ..distances import LpDistance
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_miner import BaseTupleMiner


# adapted from
# https://github.com/chaoyuaw/incubator-mxnet/blob/master/example/gluon/embedding_learning/model.py
class DistanceWeightedMiner(BaseTupleMiner):
    def __init__(self, cutoff=0.5, nonzero_loss_cutoff=1.4, **kwargs):
        super().__init__(**kwargs)
        c_f.assert_distance_type(
            self, LpDistance, p=2, power=1, normalize_embeddings=True
        )
        self.cutoff = float(cutoff)
        self.nonzero_loss_cutoff = float(nonzero_loss_cutoff)
        self.add_to_recordable_attributes(
            list_of_names=["cutoff", "nonzero_loss_cutoff"], is_stat=False
        )

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        dtype = embeddings.dtype
        d = float(embeddings.size(1))
        mat = self.distance(embeddings, ref_emb)

        # Cut off to avoid high variance.
        mat = torch.clamp(mat, min=self.cutoff)

        # See the first equation from Section 4 of the paper
        log_weights = (2.0 - d) * torch.log(mat) - ((d - 3) / 2) * torch.log(
            1.0 - 0.25 * (mat**2.0)
        )

        inf_or_nan = torch.isinf(log_weights) | torch.isnan(log_weights)

        # Sample only negative examples by setting weights of
        # the same-class examples to 0.
        mask = torch.ones_like(log_weights)
        same_class = labels.unsqueeze(1) == ref_labels.unsqueeze(0)
        mask[same_class] = 0
        log_weights = log_weights * mask
        # Subtract max(log(distance)) for stability.
        weights = torch.exp(log_weights - torch.max(log_weights[~inf_or_nan]))

        weights = (
            weights * mask * (c_f.to_dtype(mat < self.nonzero_loss_cutoff, dtype=dtype))
        )
        weights[inf_or_nan] = 0

        weights = weights / torch.sum(weights, dim=1, keepdim=True)

        return lmu.get_random_triplet_indices(
            labels, ref_labels=ref_labels, weights=weights
        )
