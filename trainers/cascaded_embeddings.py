#! /usr/bin/env python3


from . import train_with_classifier as twc


class CascadedEmbeddings(twc.TrainWithClassifier):
    def __init__(self, embedding_sizes, logit_sizes=None, **kwargs):
        super(CascadedEmbeddings, self).__init__(**kwargs)
        self.embedding_sizes = embedding_sizes
        self.logit_sizes = logit_sizes

    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        embeddings, labels = self.compute_embeddings(data, labels)
        s = 0
        for curr_size in self.embedding_sizes:
            e = embeddings[:, s : s + curr_size]
            indices_tuple = self.maybe_mine_embeddings(e, labels)
            self.losses["metric_loss"] += self.maybe_get_metric_loss(e, labels, indices_tuple)
            s += curr_size

        logits = self.maybe_get_logits(embeddings)
        if logits is not None:
            s = 0
            for curr_size in self.logit_sizes:
                L = logits[:, s : s + curr_size]
                self.losses["classifier_loss"] += self.maybe_get_classifier_loss(L, labels)
                s += curr_size