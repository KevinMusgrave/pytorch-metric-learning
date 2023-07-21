from collections import defaultdict

import torch

from .base_reducer import BaseReducer
from .mean_reducer import MeanReducer


class DefaultModuleDict(torch.nn.ModuleDict):
    def __init__(self, module_factory, modules):
        torch.nn.ModuleDict.__init__(self, modules)
        self._modules = defaultdict(module_factory, self._modules)


class MultipleReducers(BaseReducer):
    def __init__(self, reducers, default_reducer: BaseReducer = None, **kwargs):
        super().__init__(**kwargs)
        reducer_type = MeanReducer if default_reducer is None else type(default_reducer)
        self.reducers = DefaultModuleDict(
            module_factory=lambda: reducer_type(), modules=reducers
        )

    def forward(self, loss_dict, embeddings, labels):
        self.reset_stats()
        sub_losses = torch.zeros(
            len(loss_dict), dtype=embeddings.dtype, device=embeddings.device
        )

        for loss_count, (loss_name, loss_info) in enumerate(loss_dict.items()):
            input_dict = {loss_name: loss_info}
            loss_val = self.reducers[loss_name](input_dict, embeddings, labels)
            sub_losses[loss_count] = loss_val

        return self.sub_loss_reduction(sub_losses, embeddings, labels)

    def sub_loss_reduction(self, sub_losses, embeddings=None, labels=None):
        return torch.sum(sub_losses)
