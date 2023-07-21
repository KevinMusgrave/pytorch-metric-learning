import inspect
import re

from ..utils import common_functions as c_f
from ..utils.module_with_records_and_reducer import ModuleWithRecordsReducerAndDistance
from . import mixins
from .mixins import EmbeddingRegularizerMixin


class BaseMetricLossFunction(ModuleWithRecordsReducerAndDistance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.emb_loss_regularizer = EmbeddingRegularizerMixin(
            **kwargs
        )  # Avoid multiple inheritance errors. In this way if a loss function inherits from a RegularizerMixin subclass it does not affect the mro

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        """
        This has to be implemented and is what actually computes the loss.
        """
        raise NotImplementedError

    def forward(
        self, embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None
    ):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
        Returns: the loss
        """
        self.reset_stats()
        c_f.check_shapes(embeddings, labels)
        if labels is not None:
            labels = c_f.to_device(labels, embeddings)
        ref_emb, ref_labels = c_f.set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        loss_dict = self.compute_loss(
            embeddings, labels, indices_tuple, ref_emb, ref_labels
        )
        self.emb_loss_regularizer.add_embedding_regularization_to_loss_dict(
            loss_dict, embeddings
        )
        return self.reducer(loss_dict, embeddings, labels)

    def zero_loss(self):
        return {"losses": 0, "indices": None, "reduction_type": "already_reduced"}

    def zero_losses(self):
        return {loss_name: self.zero_loss() for loss_name in self.sub_loss_names()}

    def _sub_loss_names(self):
        return ["loss"]

    def sub_loss_names(self):
        return self._sub_loss_names() + self.all_regularization_loss_names()

    def all_regularization_loss_names(self):
        reg_loss_names = []
        for base_class in inspect.getmro(self.__class__):
            if base_class.__module__ == mixins.__name__:
                m = re.search(r"(\w+)RegularizerMixin", base_class.__name__)
                if m is not None:
                    reg_loss_names.append(m.group(1).lower())
        return reg_loss_names
