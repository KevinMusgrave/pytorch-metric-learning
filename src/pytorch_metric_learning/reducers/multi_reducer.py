from .base_reducer import BaseReducer
from .mean_reducer import MeanReducer
import torch

class MultipleReducers(BaseReducer):
    def __init__(self, reducers, default_reducer=None):
        super().__init__()
        self.reducers = reducers
        self.default_reducer = MeanReducer() if default_reducer is None else default_reducer

    def forward(self, loss_dict, embeddings, labels):
        sub_losses = torch.zeros(len(loss_dict)).to(embeddings.device)
        loss_count = 0
        for loss_name, loss_info in loss_dict.items():
            if loss_name in self.reducers:
                loss_val = self.reducers[loss_name]({loss_name: loss_info}, embeddings, labels)
            else:
                loss_val = self.default_reducer({loss_name: loss_info}, embeddings, labels)
            sub_losses[loss_count] = loss_val
            loss_count += 1
        return self.sub_loss_reduction(sub_losses, embeddings, labels)            
