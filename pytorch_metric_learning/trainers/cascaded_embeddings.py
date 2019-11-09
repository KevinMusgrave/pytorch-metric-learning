#! /usr/bin/env python3


from .train_with_classifier import TrainWithClassifier


class CascadedEmbeddings(TrainWithClassifier):
    def __init__(self, embedding_sizes, logit_sizes=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding_sizes = embedding_sizes
        self.logit_sizes = logit_sizes

    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        embeddings, labels = self.compute_embeddings(data, labels)
        s = 0
        for i, curr_size in enumerate(self.embedding_sizes):
            e = embeddings[:, s : s + curr_size]
            indices_tuple = self.maybe_mine_embeddings(e, labels)
            self.losses["metric_loss"] += self.maybe_get_metric_loss(e, labels, indices_tuple)
            self.update_record_keeper_in_loop(self.loss_funcs, ["metric_loss"], i)
            self.update_record_keeper_in_loop(self.mining_funcs, ["post_gradient_miner"], i)
            s += curr_size

        logits = self.maybe_get_logits(embeddings)
        if logits is not None:
            s = 0
            for i, curr_size in enumerate(self.logit_sizes):
                L = logits[:, s : s + curr_size]
                self.losses["classifier_loss"] += self.maybe_get_classifier_loss(L, labels)
                self.update_record_keeper_in_loop(self.loss_funcs, ["classifier_loss"], i)
                s += curr_size

    def update_record_keeper_in_loop(self, input_dict, keys, loop_iter):
        if self.record_keeper is not None:
            subset_dict = {}
            for k in keys:
                if k in input_dict:
                    subset_dict['%s_%d'%(k, loop_iter)] = input_dict[k]
            self.record_keeper.update_records(subset_dict, self.get_global_iteration())

    def record_these(self):
        # remove loss_funcs and mining_funcs since these are recorded in the loops above
        data_to_record = super().record_these()
        return data_to_record[:2] + data_to_record[4:]
