#! /usr/bin/env python3

from ..losses import GenericPairLoss
import torch
from ..utils import loss_and_miner_utils as lmu

class IntraPairVarianceLoss(GenericPairLoss):

    def __init__(self, pos_eps=0.01, neg_eps=0.01, **kwargs):
        super().__init__(**kwargs, use_similarity=True, mat_based_loss=False)        
        self.pos_eps = pos_eps
        self.neg_eps = neg_eps
        self.add_to_recordable_attributes(list_of_names=["pos_loss", "neg_loss"])
    
    # pos_pairs and neg_pairs already represent cos(theta)
    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        self.pos_loss, self.neg_loss = 0, 0
        if len(pos_pairs) > 0:
            mean_pos_sim = torch.mean(pos_pairs)
            pos_var = (1-self.pos_eps)*mean_pos_sim - pos_pairs
            self.pos_loss = torch.mean(torch.nn.functional.relu(pos_var)**2)
        if len(neg_pairs) > 0:
            mean_neg_sim = torch.mean(neg_pairs)
            neg_var = neg_pairs - (1+self.neg_eps)*mean_neg_sim
            self.neg_loss = torch.mean(torch.nn.functional.relu(neg_var)**2)
        return self.pos_loss+self.neg_loss

