import torch
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f
from .base_metric_loss_function import BaseMetricLossFunction
from .contrastive_loss import ContrastiveLoss
from ..reducers import AvgNonZeroReducer, MultipleReducers, MeanReducer, DoNothingReducer
from ..distances import SNRDistance

class SignalToNoiseRatioContrastiveLoss(BaseMetricLossFunction):

    def __init__(self, pos_margin, neg_margin, regularizer_weight, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(self.distance, SNRDistance), "SignalToNoiseRatioContrastiveLoss requires the distance metric to be SNRDistance"
        self.contrastive_loss = ContrastiveLoss(pos_margin=pos_margin, 
                                                neg_margin=neg_margin, 
                                                reducer=DoNothingReducer(),
                                                distance=self.distance)
        self.regularizer_weight = regularizer_weight
        self.add_to_recordable_attributes(name="feature_distance_from_zero_mean_distribution", is_stat=True)
        
    def compute_loss(self, embeddings, labels, indices_tuple):
        loss_dict = self.contrastive_loss(embeddings, labels, indices_tuple)
        self.feature_distance_from_zero_mean_distribution = torch.abs(torch.sum(embeddings, dim=1))
        reg_loss = 0
        if self.regularizer_weight > 0:
            reg_loss = self.regularizer_weight * self.feature_distance_from_zero_mean_distribution
        loss_dict["reg_loss"] = {"losses": reg_loss, "indices": c_f.torch_arange_from_size(embeddings), "reduction_type": "element"}
        return loss_dict


    def get_default_reducer(self):
        return MultipleReducers({"reg_loss": MeanReducer()}, default_reducer=AvgNonZeroReducer())

    def _sub_loss_names(self):
        return ["pos_loss", "neg_loss", "reg_loss"]

    def get_default_distance(self):
        return SNRDistance()