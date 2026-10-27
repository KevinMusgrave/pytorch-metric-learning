import torch

from ..distances import LpDistance
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_miner import BaseMiner


# adapted from
# https://github.com/chaoyuaw/incubator-mxnet/blob/master/example/gluon/embedding_learning/model.py
class DistanceWeightedMiner(BaseMiner):
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
        d = float(embeddings.size(1))
        mat = self.distance(embeddings, ref_emb)

        # Cut off to avoid high variance.
        mat = torch.clamp(mat, min=self.cutoff)

        # See the first equation from Section 4 of the paper
        log_weights = (2.0 - d) * torch.log(mat) - ((d - 3) / 2) * torch.log(
            1.0 - 0.25 * (mat**2.0)
        )

        inf_or_nan = torch.isinf(log_weights) | torch.isnan(log_weights)

        # Only eligible negatives can affect each anchor's distribution.
        same_class = labels.unsqueeze(1) == ref_labels.unsqueeze(0)
        eligible = ~same_class & ~inf_or_nan & (mat < self.nonzero_loss_cutoff)
        log_weights = log_weights.masked_fill(~eligible, -torch.inf)
        # Normalize each anchor independently to avoid cross-row underflow.
        row_max = torch.max(log_weights, dim=1, keepdim=True).values
        row_max = row_max.masked_fill(~torch.isfinite(row_max), 0)
        weights = torch.exp(log_weights - row_max)
        # Avoid overflowing the sum of large float16 reference sets.
        if weights.dtype == torch.float16:
            weights = weights.float()
        weights = weights / torch.sum(weights, dim=1, keepdim=True).clamp(min=1)

        return lmu.get_random_triplet_indices(
            labels, ref_labels=ref_labels, weights=weights
        )
