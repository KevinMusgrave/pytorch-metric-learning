import torch
import torch.nn as nn

# This is a basic multilayer perceptron
# This code is from https://github.com/KevinMusgrave/powerful_benchmarker
class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)


# This is for replacing the last layer of a pretrained network.
# This code is from https://github.com/KevinMusgrave/powerful_benchmarker
class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# This code is from https://github.com/KevinMusgrave/powerful_benchmarker
class ListOfModels(nn.Module):
    def __init__(self, list_of_models, input_sizes=None, operation_before_concat=None):
        super().__init__()
        self.list_of_models = nn.ModuleList(list_of_models)
        self.input_sizes = input_sizes
        self.operation_before_concat = (lambda x: x) if not operation_before_concat else operation_before_concat
        for k in ["mean", "std", "input_space", "input_range"]:
            setattr(self, k, getattr(list_of_models[0], k, None))

    def forward(self, x):
        outputs = []
        if self.input_sizes is None:
            for m in self.list_of_models:
                curr_output = self.operation_before_concat(m(x))
                outputs.append(curr_output)
        else:
            s = 0
            for i, y in enumerate(self.input_sizes):
                curr_input = x[:, s : s + y]
                curr_output = self.operation_before_concat(self.list_of_models[i](curr_input))
                outputs.append(curr_output)
                s += y
        return torch.cat(outputs, dim=-1)