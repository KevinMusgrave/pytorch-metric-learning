from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.utils import common_functions as c_f

from .triplet_margin_loss import TripletMarginLoss
import torch
import numpy as np
import copy
from collections import defaultdict


def concat_indices_tuple(x):
    return [torch.cat(y) for y in zip(*x)]

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
        masks, labels_list = self.create_masks_train(labels)  ## True for gallery
        masks = masks.to(embeddings.device)

        masks_float = masks.type(embeddings.type()).to(embeddings.device)
        padded = masks_float.unsqueeze(-1) * embeddings.unsqueeze(0)  # For broadcasting

        M = len(masks)
        P = len(labels_list)
        D = len(embeddings[0])
      
        centroids_mask = masks.view(M, -1, M)
        padded_tmp = padded.view(M, -1, M, D)
      
        valid_inst = centroids_mask.sum(-1)
        valid_inst_bool = centroids_mask.sum(-1).bool()
        centroids_emb = padded_tmp.sum(-2) / valid_inst.masked_fill(
            valid_inst == 0, 1
        ).unsqueeze(-1)

        embeddings_collect = []
        labels_collect = []
        tuple_indices_collect = []
        starting_idx = 0
        for i in range(M):
            if valid_inst_bool[i].sum() <= 1:
                continue

            current_mask = masks[i, :]
            current_labels = labels[~current_mask]
            query_feat = embeddings[~current_mask]
            
            current_centroids = centroids_emb[i]
            current_centroids = current_centroids[
                torch.abs(current_centroids).sum(1) > 1e-7
            ]

            embeddings_concat = torch.cat((query_feat, current_centroids))
            labels_concat = torch.cat((current_labels, current_labels))

            indices_tuple = lmu.get_all_triplets_indices(labels_concat)
            indices_tuple = [x[:len(x) // 2]+starting_idx for x in indices_tuple]
            tuple_indices_collect.append(indices_tuple)
            starting_idx += len(labels_concat)

            embeddings_collect.append(embeddings_concat)
            labels_collect.append(labels_concat)           

        indices_tuple = concat_indices_tuple(tuple_indices_collect)
        if len(indices_tuple) == 0:
            return self.zero_losses()

        final_embeddings = torch.cat(embeddings_collect)
        final_labels = torch.cat(labels_collect)


        loss = self.triplet_loss.compute_loss(final_embeddings, final_labels, indices_tuple)

        return loss

    def create_masks_train(self, class_labels):
        labels_dict = defaultdict(list)
        class_labels = class_labels.detach().cpu().numpy()
        for idx, pid in enumerate(class_labels):
            labels_dict[pid].append(idx)
        labels_list = [v for k, v in labels_dict.items()]
        labels_list_copy = copy.deepcopy(labels_list)
        lens_list = [len(item) for item in labels_list]
        lens_list_cs = np.cumsum(lens_list)

        inst_per_class = [len(item) for item in labels_dict.values()]
        
        max_gal_num = max(inst_per_class)
          ## TODO Should allow usage of all permuations
        if False in [max_gal_num == one for one in inst_per_class] and not self.allow_imbalanced:
            raise Exception("Unequal number of instances provided per label class. Set allow_imbalanced to True to surpress this warning.")


        masks = torch.ones((max_gal_num, len(class_labels)), dtype=bool)
        for _ in range(max_gal_num):
            for i, inner_list in enumerate(labels_list):
                if len(inner_list) > 0:
                    # random.shuffle(inner_list)
                    masks[_, inner_list.pop(0)] = 0
                else:
                    start_ind = lens_list_cs[i - 1]
                    end_ind = start_ind + lens_list[i]
                    masks[_, start_ind:end_ind] = 0

        return masks, labels_list_copy


    def get_default_reducer(self):
        return AvgNonZeroReducer()
