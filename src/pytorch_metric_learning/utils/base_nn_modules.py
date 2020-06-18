import torch
from . import common_functions as c_f


class ModuleWithRecords(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def add_to_recordable_attributes(self, name=None, list_of_names=None, is_stat=False):
        c_f.add_to_recordable_attributes(self, name=name, list_of_names=list_of_names, is_stat=is_stat)

    def reset_stats(self):
        c_f.reset_stats(self)


from ..reducers import MeanReducer, MultipleReducers
class ModuleWithRecordsAndReducer(ModuleWithRecords):
    def __init__(self, reducer=None, **kwargs):
        super().__init__(**kwargs)
        self.reducer = self.get_reducer() if reducer is None else reducer

    def get_default_reducer(self):
        return MeanReducer()

    def get_reducer(self):
        reducer = self.get_default_reducer()
        if isinstance(reducer, MultipleReducers) or len(self.sub_loss_names()) == 1:
            return reducer
        return MultipleReducers({k:self.get_default_reducer() for k in self.sub_loss_names()})

    def sub_loss_names(self):
        return ["loss"]