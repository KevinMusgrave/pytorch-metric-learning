import torch.nn as nn
import torch


class ListOfModels(nn.Module):
    def __init__(self, list_of_models, input_sizes=None):
        super().__init__()
        self.list_of_models = nn.ModuleList(list_of_models)
        self.input_sizes = input_sizes

    def forward(self, x):
        outputs = []
        if self.input_sizes is None:
            for m in self.list_of_models:
                outputs.append(m(x))
            return torch.cat(outputs, dim=-1)
        else:
            s = 0
            for i, y in enumerate(self.input_sizes):
                curr_input = x[:, s : s + y]
                outputs.append(self.list_of_models[i](curr_input))
                s += y
        return torch.cat(outputs, dim=-1)
