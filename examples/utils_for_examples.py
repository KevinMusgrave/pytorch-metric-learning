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


def get_record_keeper():
    # record_keeper is a useful package for logging data during training and testing
    # You can use the trainers and testers without record_keeper.
    # But if you'd like to install it, then do pip install record_keeper
    # See more info about it here https://github.com/KevinMusgrave/record_keeper
    try:
        import os
        import errno
        import record_keeper as record_keeper_package
        from torch.utils.tensorboard import SummaryWriter

        def makedir_if_not_there(dir_name):
            try:
                os.makedirs(dir_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        pkl_folder = "example_logs"
        tensorboard_folder = "example_tensorboard"
        makedir_if_not_there(pkl_folder)
        makedir_if_not_there(tensorboard_folder)
        pickler_and_csver = record_keeper_package.PicklerAndCSVer(pkl_folder)
        tensorboard_writer = SummaryWriter(log_dir=tensorboard_folder)
        return record_keeper_package.RecordKeeper(tensorboard_writer, pickler_and_csver, ["record_these", "learnable_param_names"])

    except ModuleNotFoundError:
        return None