from .metric_loss_only import MetricLossOnly
import logging
from ..utils import common_functions as c_f
import torch

class UnsupervisedEmbeddingsUsingAugmentations(MetricLossOnly):
    def __init__(self, transforms, **kwargs):
        super().__init__(**kwargs)
        self.label_mapper = lambda label, hierarchy_level: label
        self.collate_fn = self.get_custom_collate_fn(transforms, self.possible_data_keys)
        self.initialize_dataloader()
        logging.info("Transforms: %s"%transforms)

    def get_custom_collate_fn(self, transforms, possible_data_keys):
        def custom_collate_fn(data):
            transformed_data, labels = [], []
            for i, d in enumerate(data):
                img = c_f.try_keys(d, possible_data_keys)
                for t in transforms:
                    transformed_data.append(t(img))
                    labels.append(i)
            return {possible_data_keys[0]: torch.stack(transformed_data, dim=0), "label": torch.LongTensor(labels)}
        return custom_collate_fn
