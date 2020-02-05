from .metric_loss_only import MetricLossOnly
import logging
from ..utils import common_functions as c_f
import torch

class UnsupervisedEmbeddingsUsingAugmentations(MetricLossOnly):
    def __init__(self, transforms, data_and_label_setter=None, **kwargs):
        super().__init__(**kwargs)
        self.data_and_label_setter = data_and_label_setter
        self.initialize_data_and_label_setter()
        self.collate_fn = self.get_custom_collate_fn(transforms)
        self.initialize_dataloader()
        logging.info("Transforms: %s"%transforms)


    def get_custom_collate_fn(self, transforms, data_and_label_setter):
        def custom_collate_fn(data):
            transformed_data, labels = [], []
            for i, d in enumerate(data):
                img, _ = self.data_and_label_getter(d)
                for t in transforms:
                    transformed_data.append(t(img))
                    labels.append(i)
            return self.data_and_label_setter((torch.stack(transformed_data, dim=0), torch.LongTensor(labels)))
        return custom_collate_fn

    def initialize_data_and_label_setter(self):
        if self.data_and_label_setter is None:
            def data_and_label_setter(x):
                return x
            self.data_and_label_setter = data_and_label_setter