#! /usr/bin/env python3

from . import metric_loss_only as mlo


class TrainWithClassifier(mlo.MetricLossOnly):
    def loss_names(self):
        return ["metric_loss", "classifier_loss"]

    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        embeddings, labels = self.compute_embeddings(data, labels)
        logits = self.maybe_get_logits(embeddings)
        indices_tuple = self.maybe_mine_embeddings(embeddings, labels)
        self.losses["metric_loss"] = self.maybe_get_metric_loss(embeddings, labels, indices_tuple)
        self.losses["classifier_loss"] = self.maybe_get_classifier_loss(logits, labels)

    def maybe_get_classifier_loss(self, logits, labels):
        if logits is not None:
            return self.loss_funcs["classifier_loss"](logits, labels.to(logits.device))
        return 0

    def maybe_get_logits(self, embeddings):
        if self.loss_weights.get("classifier_loss",0) > 0:
            return self.models["classifier"](embeddings)
        return None

        
        
        
        
