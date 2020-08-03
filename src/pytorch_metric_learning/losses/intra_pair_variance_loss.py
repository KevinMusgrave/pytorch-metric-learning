from ..losses import GenericPairLoss
import torch
from ..utils import loss_and_miner_utils as lmu
from ..distances import CosineSimilarity

class IntraPairVarianceLoss(GenericPairLoss):

    def __init__(self, pos_eps=0.01, neg_eps=0.01, **kwargs):
        super().__init__(mat_based_loss=False, **kwargs)        
        self.pos_eps = pos_eps
        self.neg_eps = neg_eps
        self.add_to_recordable_attributes(list_of_names=["pos_eps", "neg_eps"], is_stat=False)

    # pos_pairs and neg_pairs already represent cos(theta)
    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        pos_loss, neg_loss = 0, 0
        if len(pos_pairs) > 0:
            mean_pos_sim = torch.mean(pos_pairs)
            pos_var = self.variance_with_eps(pos_pairs, mean_pos_sim, self.pos_eps, self.distance.is_inverted)
            pos_loss = torch.nn.functional.relu(pos_var)**2
        if len(neg_pairs) > 0:
            mean_neg_sim = torch.mean(neg_pairs)
            neg_var = self.variance_with_eps(neg_pairs, mean_neg_sim, self.neg_eps, not self.distance.is_inverted)
            neg_loss = torch.nn.functional.relu(neg_var)**2
        pos_pairs_idx = lmu.pos_pairs_from_tuple(indices_tuple)
        neg_pairs_idx = lmu.neg_pairs_from_tuple(indices_tuple)
        return {"pos_loss": {"losses": pos_loss, "indices": pos_pairs_idx, "reduction_type": "pos_pair"}, 
                "neg_loss": {"losses": neg_loss, "indices": neg_pairs_idx, "reduction_type": "neg_pair"}}


    def variance_with_eps(self, pairs, mean_sim, eps, incentivize_increase):
        if incentivize_increase:
            return (1-eps)*mean_sim - pairs
        return pairs - (1+eps)*mean_sim


    def sub_loss_names(self):
        return ["pos_loss", "neg_loss"]

    def get_default_distance(self):
        return CosineSimilarity()
