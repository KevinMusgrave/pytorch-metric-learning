import torch
from ..utils import common_functions as c_f
from ..utils.module_with_records_and_reducer import ModuleWithRecordsReducerAndDistance


class BaseRegularizer(ModuleWithRecordsReducerAndDistance):
    def compute_loss(self, x):
        raise NotImplementedError

    def forward(self, x):
        """
        x should have shape (N, embedding_size)
        """
        self.reset_stats()
        loss_dict = self.compute_loss(x)
        return self.reducer(loss_dict, x, c_f.torch_arange_from_size(x))
