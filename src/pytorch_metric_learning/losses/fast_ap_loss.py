import torch
from .base_metric_loss_function import BaseMetricLossFunction
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f
from ..distances import LpDistance

class FastAPLoss(BaseMetricLossFunction):
    def __init__(self, num_bins=10, **kwargs):
        super().__init__(**kwargs)
        c_f.assert_distance_type(self, LpDistance, p=2)
        self.num_bins = int(num_bins)
        self.num_edges = self.num_bins + 1
        self.add_to_recordable_attributes(list_of_names=["num_bins"], is_stat=False)

    """
    Adapted from https://github.com/kunhe/FastAP-metric-learning
    """
    def compute_loss(self, embeddings, labels, indices_tuple):
        dtype, device = embeddings.dtype, embeddings.device
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        N = labels.size(0)
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels)
        I_pos = torch.zeros(N, N, dtype=dtype).to(device)
        I_neg = torch.zeros(N, N, dtype=dtype).to(device)
        I_pos[a1_idx, p_idx] = 1
        I_neg[a2_idx, n_idx] = 1
        N_pos = torch.sum(I_pos, dim=1)
        safe_N = (N_pos > 0)
        if torch.sum(safe_N) == 0:
            return self.zero_losses()
        dist_mat = self.distance(embeddings)

        histogram_max = 2**self.distance.power
        histogram_delta = histogram_max / self.num_bins
        mid_points = torch.linspace(0., histogram_max, steps=self.num_edges).view(-1,1,1).to(device).type(dtype)
        pulse = torch.nn.functional.relu(1 - torch.abs(dist_mat-mid_points)/histogram_delta)
        pos_hist = torch.t(torch.sum(pulse * I_pos, dim=2))
        neg_hist = torch.t(torch.sum(pulse * I_neg, dim=2))

        total_pos_hist = torch.cumsum(pos_hist, dim=1)
        total_hist = torch.cumsum(pos_hist + neg_hist, dim=1)
        
        h_pos_product = pos_hist * total_pos_hist
        safe_H = (h_pos_product > 0) & (total_hist > 0)
        if torch.sum(safe_H) > 0:
            FastAP = torch.zeros_like(pos_hist).to(device)
            FastAP[safe_H] = h_pos_product[safe_H] / total_hist[safe_H]
            FastAP = torch.sum(FastAP, dim=1)
            FastAP = FastAP[safe_N] / N_pos[safe_N]
            FastAP = (1-FastAP)*miner_weights[safe_N]
            return {"loss": {"losses": FastAP, "indices": torch.where(safe_N)[0], "reduction_type": "element"}}
        return self.zero_losses()
        
    def get_default_distance(self):
        return LpDistance(power=2)

