from ..reducers import MeanReducer, MultipleReducers
from .module_with_records import ModuleWithRecords

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