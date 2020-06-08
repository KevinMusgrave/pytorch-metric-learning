#! /usr/bin/env python3


from .base_trainer import BaseTrainer
from ..utils import common_functions as c_f, loss_and_miner_utils as lmu
import logging
import torch

class TwoStreamMetricLoss(BaseTrainer):

    def calculate_loss(self, curr_batch):
        (anchors, posnegs), labels = curr_batch
        embeddings = (self.compute_embeddings(anchors), self.compute_embeddings(posnegs))

        indices_tuple = self.maybe_mine_embeddings(embeddings, labels)
        self.losses["metric_loss"] = self.maybe_get_metric_loss(embeddings, labels, indices_tuple)
    
    def get_batch(self):
        self.dataloader_iter, curr_batch = c_f.try_next_on_generator(self.dataloader_iter, self.dataloader)
        anchors, posnegs, labels = self.data_and_label_getter(curr_batch)
        data = (anchors,posnegs)
        labels = c_f.process_label(labels, self.label_hierarchy_level, self.label_mapper)
        return self.maybe_do_batch_mining(data, labels)

    def maybe_get_metric_loss(self, embeddings, labels, indices_tuple):
        if self.loss_weights.get("metric_loss", 0) > 0:
            current_batch_size = embeddings[0].shape[0]
            indices_tuple = c_f.shift_indices_tuple(indices_tuple, current_batch_size)
            all_labels = torch.cat([labels, labels], dim=0)
            all_embeddings = torch.cat(embeddings, dim=0)
            return self.loss_funcs["metric_loss"](all_embeddings, all_labels, indices_tuple)
        return 0

    def maybe_mine_embeddings(self, embeddings, labels):
        # for both get_all_triplets_indices and mining_funcs
        # we need to clone labels and pass them as ref_labels 
        # to ensure triplets are generated between anchors and posnegs
        if "tuple_miner" in self.mining_funcs:
            (anchors_embeddings, posnegs_embeddings) = embeddings
            return self.mining_funcs["tuple_miner"](anchors_embeddings, labels, posnegs_embeddings, labels.clone())
        else:
            return lmu.get_all_triplets_indices(labels, labels.clone())

    def allowed_mining_funcs_keys(self):
        return ["tuple_miner"]