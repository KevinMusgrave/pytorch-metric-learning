import torch
import torch.nn.functional as F

from ..distances import CosineSimilarity
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .base_metric_loss_function import BaseMetricLossFunction


class SmoothAPLoss(BaseMetricLossFunction):
    """
    Implementation of the SmoothAP loss: https://arxiv.org/abs/2007.12163
    """

    def __init__(self, temperature=0.01, **kwargs):
        super().__init__(**kwargs)
        c_f.assert_distance_type(self, CosineSimilarity)
        self.temperature = temperature

    def get_default_distance(self):
        return CosineSimilarity()

    # Implementation is based on the original repository:
    # https://github.com/Andrew-Brown1/Smooth_AP/blob/master/src/Smooth_AP_loss.py#L87
    def compute_loss(self, embeddings, labels, iices_tuple, ref_emb, ref_labels):
        # The loss expects labels such that there is the same number of elements for each class
        # The number of classes is not important, nor their order, but the number of elements must be the same, eg.
        #
        # The following label is valid:
        # [ A,A,A, B,B,B, C,C,C ]
        # The following label is NOT valid:
        # [ B,B,B  A,A,A,A,  C,C,C ]
        #
        counts = torch.bincount(labels)
        nonzero_indices = torch.nonzero(counts, as_tuple=True)[0]
        nonzero_counts = counts[nonzero_indices]
        if nonzero_counts.unique().size(0) != 1:
            raise ValueError(
                "All classes must have the same number of elements in the labels.\n"
                "The given labels have the following number of elements: {}.\n"
                "You can achieve this using the samplers.MPerClassSampler class and setting the batch_size and m.".format(
                    nonzero_counts.cpu().tolist()
                )
            )

        batch_size = embeddings.size(0)
        num_classes_batch = batch_size // torch.unique(labels).size(0)

        mask = 1.0 - torch.eye(batch_size)
        mask = mask.unsqueeze(dim=0).repeat(batch_size, 1, 1)

        sims = F.cosine_similarity(
            embeddings[None, :, :], embeddings[:, None, :], dim=-1
        )
        sims_repeat = sims.unsqueeze(dim=1).repeat(1, batch_size, 1)
        sims_diff = sims_repeat - sims_repeat.permute(0, 2, 1)
        sims_sigm = F.sigmoid(sims_diff / self.temperature) * mask.to(sims_diff.device)
        sims_ranks = torch.sum(sims_sigm, dim=-1) + 1

        xs = embeddings.view(
            num_classes_batch, batch_size // num_classes_batch, embeddings.size(-1)
        ).permute(0, 2, 1)
        pos_mask = 1.0 - torch.eye(batch_size // num_classes_batch)
        pos_mask = (
            pos_mask.unsqueeze(dim=0)
            .unsqueeze(dim=0)
            .repeat(num_classes_batch, batch_size // num_classes_batch, 1, 1)
        )

        sims_pos = F.cosine_similarity(xs[:, :, None, :], xs[:, :, :, None])
        sims_pos_repeat = sims_pos.unsqueeze(dim=2).repeat(
            1, 1, batch_size // num_classes_batch, 1
        )
        sims_pos_diff = sims_pos_repeat - sims_pos_repeat.permute(0, 1, 3, 2)

        sims_pos_sigm = F.sigmoid(sims_pos_diff / self.temperature) * pos_mask.to(
            sims_diff.device
        )
        sims_pos_ranks = torch.sum(sims_pos_sigm, dim=-1) + 1

        ap = torch.zeros(1).to(embeddings.device)
        g = batch_size // num_classes_batch
        for i in range(num_classes_batch):
            pos_divide = torch.sum(
                sims_pos_ranks[i] / sims_ranks[i * g : (i + 1) * g, i * g : (i + 1) * g]
            )
            ap = ap + (pos_divide / g) / batch_size

        loss = 1 - ap
        return {
            "loss": {
                "losses": loss,
                "indices": None,
                "reduction_type": "already_reduced",
            }
        }
