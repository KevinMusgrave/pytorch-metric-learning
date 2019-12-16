from pytorch_metric_learning import losses, miners, trainers
import numpy as np
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim
import logging
logging.getLogger().setLevel(logging.INFO)

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
    record_keeper = record_keeper_package.RecordKeeper(tensorboard_writer, pickler_and_csver, ["record_these", "learnable_param_names"])

except ModuleNotFoundError:
    record_keeper = None


##############################
########## Training ##########
##############################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set trunk model and replace the softmax layer with an identity function
trunk = models.resnet18(pretrained=True)
trunk_output_size = trunk.fc.in_features
trunk.fc = Identity()
trunk = torch.nn.DataParallel(trunk.to(device))

# Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
embedder = torch.nn.DataParallel(MLP([trunk_output_size, 64]).to(device))

# Set optimizers
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.00001, weight_decay=0.00005)
embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=0.00001, weight_decay=0.00005)

# Set the image transform
img_transform = transforms.Compose([transforms.Resize(256),
									transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=227),
									transforms.RandomHorizontalFlip(0.5),
									transforms.ToTensor(),
									transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Set the datasets
train_dataset = datasets.CIFAR100(root="CIFAR100_Dataset", train=True, transform=img_transform, download=True)
val_dataset = datasets.CIFAR100(root="CIFAR100_Dataset", train=False, transform=img_transform, download=True)

# Set the loss function
loss = losses.TripletMarginLoss(margin=0.01)

# Set the mining function
miner = miners.MultiSimilarityMiner(epsilon=0.1)

# Set other training parameters
batch_size = 128
num_epochs = 2
iterations_per_epoch = 100

# Package the above stuff into dictionaries.
models = {"trunk": trunk, "embedder": embedder}
optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer}
loss_funcs = {"metric_loss": loss}
mining_funcs = {"post_gradient_miner": miner}

trainer = trainers.MetricLossOnly(models,
								optimizers,
								batch_size,
								loss_funcs,
								mining_funcs,
								num_epochs,
								iterations_per_epoch,
								train_dataset,
                                record_keeper=record_keeper)

trainer.train()




#############################
########## Testing ##########
############################# 

# The testing module requires faiss and scikit-learn
# So if you don't have these, then this import will break
from pytorch_metric_learning import testers

tester = testers.GlobalEmbeddingSpaceTester(record_keeper=record_keeper)
dataset_dict = {"train": train_dataset, "val": val_dataset}
epoch = 2

tester.test(dataset_dict, epoch, trunk, embedder)

if record_keeper is not None:
    record_keeper.pickler_and_csver.save_records()