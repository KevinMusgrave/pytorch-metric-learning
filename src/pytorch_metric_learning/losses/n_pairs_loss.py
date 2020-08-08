from .base_metric_loss_function import BaseMetricLossFunction
import torch
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f
from ..distances import DotProductSimilarity

class NPairsLoss(BaseMetricLossFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_to_recordable_attributes(name="num_pairs", is_stat=True)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def compute_loss(self, embeddings, labels, indices_tuple):
        anchor_idx, positive_idx = lmu.convert_to_pos_pairs_with_unique_labels(indices_tuple, labels)
        self.num_pairs = len(anchor_idx)
        if self.num_pairs == 0:
            return self.zero_losses()
        anchors, positives = embeddings[anchor_idx], embeddings[positive_idx]
        targets = torch.arange(self.num_pairs).to(embeddings.device)
        sim_mat = self.distance(anchors, positives)
        if not self.distance.is_inverted:
            sim_mat = -sim_mat
        return {"loss": {"losses": self.cross_entropy(sim_mat, targets), "indices": anchor_idx, "reduction_type": "element"}}

    def get_default_distance(self):
        return DotProductSimilarity()