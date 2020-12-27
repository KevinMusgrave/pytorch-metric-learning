import logging

import torch

from ..utils import common_functions as c_f
from .metric_loss_only import MetricLossOnly


class UnsupervisedEmbeddingsUsingAugmentations(MetricLossOnly):
    def __init__(self, transforms, data_and_label_setter=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_and_label_setter = data_and_label_setter
        self.initialize_data_and_label_setter()
        self.transforms = transforms
        self.collate_fn = self.custom_collate_fn
        self.initialize_dataloader()
        logging.info("Transforms: %s" % transforms)

    def initialize_data_and_label_setter(self):
        if self.data_and_label_setter is None:
            self.data_and_label_setter = c_f.return_input

    def custom_collate_fn(self, data):
        transformed_data, labels = [], []
        for i, d in enumerate(data):
            img, _ = self.data_and_label_getter(d)
            for t in self.transforms:
                transformed_data.append(t(img))
                labels.append(i)
        return self.data_and_label_setter(
            (torch.stack(transformed_data, dim=0), torch.LongTensor(labels))
        )
