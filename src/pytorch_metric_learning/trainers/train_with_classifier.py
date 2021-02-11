from ..utils import common_functions as c_f
from .metric_loss_only import MetricLossOnly


class TrainWithClassifier(MetricLossOnly):
    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        embeddings = self.compute_embeddings(data)
        logits = self.maybe_get_logits(embeddings)
        indices_tuple = self.maybe_mine_embeddings(embeddings, labels)
        self.losses["metric_loss"] = self.maybe_get_metric_loss(
            embeddings, labels, indices_tuple
        )
        self.losses["classifier_loss"] = self.maybe_get_classifier_loss(logits, labels)

    def maybe_get_classifier_loss(self, logits, labels):
        if logits is not None:
            return self.loss_funcs["classifier_loss"](
                logits, c_f.to_device(labels, logits)
            )
        return 0

    def maybe_get_logits(self, embeddings):
        if (
            self.models.get("classifier", None)
            and self.loss_weights.get("classifier_loss", 0) > 0
        ):
            return self.models["classifier"](embeddings)
        return None

    def allowed_model_keys(self):
        return super().allowed_model_keys() + ["classifier"]

    def allowed_loss_funcs_keys(self):
        return super().allowed_loss_funcs_keys() + ["classifier_loss"]
