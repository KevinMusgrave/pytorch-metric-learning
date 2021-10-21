from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from .triplet_margin_loss import TripletMarginLoss
import torch
import numpy as np
import copy
from collections import defaultdict

class CentroidTripletLoss(BaseMetricLossFunction):
    def __init__(
        self,
        allow_imbalanced=False,
        margin=0.05,
        swap=False,
        smooth_loss=False,
        triplets_per_anchor="all",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.allow_imbalanced = allow_imbalanced
        self.margin = margin
        self.swap = swap
        self.smooth_loss = smooth_loss
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)

        self.triplet_loss = TripletMarginLoss(
            margin=margin,
            swap=swap,
            smooth_loss=smooth_loss,
            triplets_per_anchor=triplets_per_anchor,
            **kwargs
        )

    def compute_loss(self, embeddings, labels, indices_tuple=None):
        '''
        "During training stage each mini-batch contains 𝑃 distinct item
        classes with 𝑀 samples per class, resulting in batch size of 𝑃 × 𝑀."
        '''
        masks, labels_list, query_indices = self.create_masks_train(labels)
        P = len(labels_list)
        M = max([len(instances) for instances in labels_list])
        DIM = embeddings.size(-1)

        '''
        "...each sample from S𝑘 is used as a query 𝑞𝑘 and the rest 
        𝑀 −1 samples are used to build a prototype centroid"
        i.e. for each class k of M items, we make M pairs of (query, centroid),
        making a total of P*M total pairs.

        masks = (M*P x len(embeddings)) matrix
        labels_list[i] = indicies of embeddings belonging to ith class

        centroids_emd.shape == (M*P, DIM)
        i.e.    centroids_emb[0] == centroid vector for 0th class, where the first embedding is the query vector
                centroids_emb[1] == centroid vector for 0th class, where the second embedding is the query vector
                centroids_emb[M+1] == centroid vector for 1th class, where the first embedding is the query vector
        '''
      
        masks = masks.to(embeddings.device)
        masks_float = masks.float().to(embeddings.device)
        inst_counts = masks_float.sum(-1)
        valid_mask = inst_counts > 0
        padded = masks_float.unsqueeze(-1) * embeddings.unsqueeze(0) 
        centroids_emb = padded.sum(-2) / inst_counts.masked_fill(
            inst_counts == 0, 1
        ).unsqueeze(-1)

        query_indices = torch.tensor(query_indices).to(embeddings.device)
        query_embeddings = embeddings.index_select(0, query_indices)
        query_labels = labels.index_select(0, query_indices)
        assert centroids_emb.size() == (M*P, DIM)
        assert query_embeddings.size() == (M*P, DIM)


        # print(query_indices)
        # print(query_embeddings)
        # print(query_labels)

        query_indices = query_indices.view((P, M)).transpose(0, 1)
        query_embeddings = query_embeddings.view((P, M, -1)).transpose(0, 1)
        query_labels = query_labels.view((P, M)).transpose(0, 1)
        centroids_emb = centroids_emb.view((P, M, -1)).transpose(0, 1)
        valid_mask = valid_mask.view((P, M)).transpose(0, 1)

        loss_collect = []
        for inst_idx in range(M):
            one_mask = valid_mask[inst_idx]
            if torch.sum(one_mask) > 1:
                one_queries = query_embeddings[inst_idx][one_mask]
                one_centroids = centroids_emb[inst_idx][one_mask]
                one_labels = query_labels[inst_idx][one_mask]
                
                combined_embeddings = torch.cat((one_queries, one_centroids))
                combined_labels = torch.cat((one_labels, one_labels))

                one_loss = self.triplet_loss(combined_embeddings, combined_labels)
                loss_collect.append(one_loss)
        loss_mean = torch.mean(torch.stack(loss_collect))

        return {
            "loss": {
                "losses": loss_mean,
                "reduction_type": "centroid",
            }
        }


    @staticmethod
    def smart_sort(x, permutation):
        d1, d2 = x.size()
        ret = x[
            torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
            permutation.flatten()
        ].view(d1, d2)
        return ret


    def create_masks_train(self, class_labels):
        labels_dict = defaultdict(list)
        class_labels = class_labels.detach().cpu().numpy()
        for idx, pid in enumerate(class_labels):
            labels_dict[pid].append(idx)

        unique_classes = list(labels_dict.keys())
        labels_list = list(labels_dict.values())

        # labels_list = [v for k, v in labels_dict.items()]
        labels_list_copy = copy.deepcopy(labels_list)
        lens_list = [len(item) for item in labels_list]
        lens_list_cs = np.cumsum(lens_list)

        M = max(len(instances) for instances in labels_list)  
        P = len(unique_classes)

        query_indices = []
        masks = torch.zeros((M * P, len(class_labels)), dtype=bool)
        for class_idx, class_insts in enumerate(labels_list):
            for instance_idx in range(M):
                matrix_idx = class_idx * M + instance_idx
                if instance_idx < len(class_insts):
                    query_indices.append(class_insts[instance_idx])
                    ones = class_insts[:instance_idx] + class_insts[instance_idx+1:]
                    masks[matrix_idx, ones] = 1
                else:
                    if self.allow_imbalanced:
                        query_indices.append(0)
                    else:
                        raise Exception("Found uneven distribution of embeddings per label. "
                        "Set allowed_imbalanced to True to surpress this error.")
        return masks, labels_list, query_indices


    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def get_default_distance(self):
        return CosineSimilarity()

    def _sub_loss_names(self):
        return ["loss1", "loss2", "loss3"]