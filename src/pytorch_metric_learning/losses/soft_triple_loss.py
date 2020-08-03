from .base_metric_loss_function import BaseMetricLossFunction
from .regularizer_mixins import WeightRegularizerMixin
from ..utils import loss_and_miner_utils as lmu, common_functions as c_f
import math
import torch
import torch.nn.functional as F
from ..distances import CosineSimilarity

###### modified from https://github.com/idstcv/SoftTriple/blob/master/loss/SoftTriple.py ######
###### Original code is Copyright@Alibaba Group ######
###### ICCV'19: "SoftTriple Loss: Deep Metric Learning Without Triplet Sampling" ######
class SoftTripleLoss(WeightRegularizerMixin, BaseMetricLossFunction):
    def __init__(self, embedding_size, num_classes, centers_per_class, la=20, gamma=0.1, margin=0.01, **kwargs):
        super().__init__(**kwargs)
        assert self.distance.is_inverted
        self.la = la
        self.gamma = 1./gamma
        self.margin = margin
        self.num_classes = num_classes
        self.centers_per_class = centers_per_class
        self.fc = torch.nn.Parameter(torch.Tensor(embedding_size, num_classes * centers_per_class))
        torch.nn.init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        self.add_to_recordable_attributes(list_of_names=["la", "gamma", "margin", "centers_per_class", "num_classes", "embedding_size"], is_stat=False)

    def cast_types(self, dtype, device):
        self.fc.data = self.fc.data.to(device).type(dtype)

    def compute_loss(self, embeddings, labels, indices_tuple):
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        sim_to_centers = self.distance(embeddings, self.fc.t())
        sim_to_centers = sim_to_centers.view(-1, self.num_classes, self.centers_per_class)
        prob = F.softmax(sim_to_centers*self.gamma, dim=2)
        sim_to_classes = torch.sum(prob*sim_to_centers, dim=2)
        margin = torch.zeros(sim_to_classes.shape, dtype=dtype).to(embeddings.device)
        margin[torch.arange(0, margin.shape[0]), labels] = self.margin
        loss = F.cross_entropy(self.la*(sim_to_classes-margin), labels, reduction='none')
        loss = loss*miner_weights
        loss_dict = {"loss": {"losses": loss, "indices": c_f.torch_arange_from_size(embeddings), "reduction_type": "element"}}
        self.add_weight_regularization_to_loss_dict(loss_dict, self.fc.t())
        return loss_dict

    def get_default_distance(self):
        return CosineSimilarity()