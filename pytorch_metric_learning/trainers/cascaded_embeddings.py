#! /usr/bin/env python3


from .base_trainer import BaseTrainer


class CascadedEmbeddings(BaseTrainer):
    def __init__(self, embedding_sizes, **kwargs):
        super().__init__(**kwargs)
        self.embedding_sizes = embedding_sizes

    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        embeddings, labels = self.compute_embeddings(data, labels)
        s = 0
        logits = []
        for i, curr_size in enumerate(self.embedding_sizes):
            curr_loss_name = "metric_loss_%d"%i 
            curr_miner_name = "post_gradient_miner_%d"%i
            curr_classifier_name = "classifier_%d"%i

            e = embeddings[:, s : s + curr_size]
            indices_tuple = self.maybe_mine_embeddings(e, labels, curr_miner_name)
            self.losses[curr_loss_name] += self.maybe_get_metric_loss(e, labels, indices_tuple, curr_loss_name)
            logits.append(self.maybe_get_logits(e, curr_classifier_name))
            s += curr_size

        for i, L in enumerate(logits):
            if L is None:
                continue
            curr_loss_name = "classifier_loss_%d"%i
            self.losses[curr_loss_name] += self.maybe_get_classifier_loss(L, labels, curr_loss_name)

    def maybe_get_metric_loss(self, embeddings, labels, indices_tuple, curr_loss_name):
        if self.loss_weights.get(curr_loss_name, 0) > 0:
            return self.loss_funcs[curr_loss_name](embeddings, labels, indices_tuple)
        return 0

    def maybe_mine_embeddings(self, embeddings, labels, curr_miner_name):
        if curr_miner_name in self.mining_funcs:
            return self.mining_funcs[curr_miner_name](embeddings, labels)
        return None

    def maybe_get_logits(self, embeddings, curr_classifier_name):
        if self.models.get(curr_classifier_name, None):
            return self.models[curr_classifier_name](embeddings)
        return None

    def maybe_get_classifier_loss(self, logits, labels, curr_loss_name):
        if self.loss_weights.get(curr_loss_name, 0) > 0:
            return self.loss_funcs[curr_loss_name](logits, labels.to(logits.device))
        return 0