from .base_metric_loss_function import BaseMetricLossFunction
from ..utils import loss_and_miner_utils as lmu
import math
import torch
import torch.nn.functional as F

###### modified from https://github.com/idstcv/SoftTriple/blob/master/loss/SoftTriple.py ######
###### Original code is Copyright@Alibaba Group ######
###### ICCV'19: "SoftTriple Loss: Deep Metric Learning Without Triplet Sampling" ######
class SoftTripleLoss(BaseMetricLossFunction):
    def __init__(self, embedding_size, num_classes, centers_per_class, la=20, gamma=0.1, reg_weight=0.2, margin=0.01, **kwargs):
        super().__init__(**kwargs)
        self.la = la
        self.gamma = 1./gamma
        self.reg_weight = reg_weight
        self.margin = margin
        self.num_classes = num_classes
        self.centers_per_class = centers_per_class
        self.total_num_centers = num_classes * centers_per_class
        self.fc = torch.nn.Parameter(torch.Tensor(embedding_size, self.total_num_centers))
        self.set_class_masks(num_classes, centers_per_class)
        torch.nn.init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        self.add_to_recordable_attributes(list_of_names=["same_class_center_similarity", "diff_class_center_similarity"])

    def compute_loss(self, embeddings, labels, indices_tuple):
        miner_weights = lmu.convert_to_weights(indices_tuple, labels)
        centers = F.normalize(self.fc, p=2, dim=0) if self.normalize_embeddings else self.fc
        sim_to_centers = torch.matmul(embeddings, centers)
        sim_to_centers = sim_to_centers.view(-1, self.num_classes, self.centers_per_class)
        prob = F.softmax(sim_to_centers*self.gamma, dim=2)
        sim_to_classes = torch.sum(prob*sim_to_centers, dim=2)
        margin = torch.zeros(sim_to_classes.shape).to(embeddings.device)
        margin[torch.arange(0, margin.shape[0]), labels] = self.margin
        loss = F.cross_entropy(self.la*(sim_to_classes-margin), labels, reduction='none')
        loss = torch.mean(loss*miner_weights)

        #regularization which encourages the centers of a class to be close to each other
        reg = 0
        center_similarities = centers.t().matmul(centers)
        if self.reg_weight > 0 and self.centers_per_class > 1:
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*center_similarities[self.same_class_mask]))/(torch.sum(self.same_class_mask))
        self.set_stats(center_similarities)
        return loss+self.reg_weight*reg

    def set_class_masks(self, num_classes, centers_per_class):
        self.diff_class_mask = torch.ones(self.total_num_centers, self.total_num_centers, dtype=torch.bool)
        if centers_per_class > 1:
            self.same_class_mask = torch.zeros(self.total_num_centers, self.total_num_centers, dtype=torch.bool)
        for i in range(num_classes):
            s, e = i*centers_per_class, (i+1)*centers_per_class
            if centers_per_class > 1:
                curr_block = torch.ones(centers_per_class, centers_per_class)
                curr_block = torch.triu(curr_block, diagonal=1)
                self.same_class_mask[s:e, s:e] = curr_block
            self.diff_class_mask[s:e, s:e] = 0

    def set_stats(self, center_similarities):
        if self.centers_per_class > 1:
            self.same_class_center_similarity = torch.mean(center_similarities[self.same_class_mask])
        self.diff_class_center_similarity = torch.mean(center_similarities[self.diff_class_mask])
