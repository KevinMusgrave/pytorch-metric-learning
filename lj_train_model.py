# Script version of LJCV-MetricLossOnly.ipynb

import logging
import matplotlib.pyplot as plt
import numpy as np
import record_keeper
import torch
import sys, os
import torch.nn as nn
import umap
from cycler import cycler
from PIL import Image
from torchvision import datasets, transforms

import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

sys.path.append(os.path.join('/layerjot', 'pytorch-image-models'))
from timm.models import create_model, list_models
from efficientnet_pytorch import EfficientNet

logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s" % pytorch_metric_learning.__version__)

# SETTINGS

effnet = list_models("efficientnet_b4")[0]
num_classes = 827 # from pretraining
output_dim = 1792
input_dim_resize = 650
input_dim_crop = 600
embedding_dim = 128
# input_dim_resize = 64
# input_dim_crop = 64

# Set other training parameters
batch_size = 24
num_epochs = 20
margin = 0.1
m_per_class = 2
eval_batch_size = 32
eval_k="max_bin_count"
patience=3
lr=0.001
# eval_k=10
# Need to run eval on the CPU because training holds onto GPU memory
eval_device = torch.device("cpu")

# NOTE: I don't think these params are going to do the right thing - revisit this choice
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
traindir = "/data/n1_crops_dataset/feather/train"
testdir = "/data/n1_crops_dataset/feather/test"
pretrained=True
CHECKPOINT = "/data/sa_models/metric_dim_64/trunk_best11.pth"

# EMBEDDER

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
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)

# TRUNK

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set trunk model and replace the softmax layer with an identity function

# For using TIMM models
# trunk = create_model(effnet, num_classes=num_classes, pretrained=pretrained)
# trunk.reset_classifier(0)

# Using large pretrained efficientnet
trunk = EfficientNet.from_pretrained("efficientnet-b7", output_dim)
num_ftrs = trunk._fc.in_features
trunk._fc = nn.Linear(num_ftrs, output_dim)
trunk.set_swish(memory_efficient=False)

# checkpoint = torch.load(CHECKPOINT)
# trunk.load_state_dict(checkpoint)
# Set classification head to identity
trunk_output_size = output_dim
trunk = torch.nn.DataParallel(trunk.to(device))

# Set embedder model. This takes in the output of the trunk and outputs the embedding dimension
embedder = torch.nn.DataParallel(MLP([trunk_output_size, trunk_output_size/2, embedding_dim]).to(device))

# Set optimizers
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.00001, weight_decay=0.0001)
embedder_optimizer = torch.optim.Adam(
    embedder.parameters(), lr=lr, weight_decay=0.0001
)

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.Resize(input_dim_resize), # Remove this when using EfficientNet - or test with it
        transforms.RandomResizedCrop(input_dim_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

val_dataset = datasets.ImageFolder(testdir, transforms.Compose([
        transforms.Resize(input_dim_resize),
        transforms.CenterCrop(input_dim_crop),
        transforms.ToTensor(),
        normalize,
    ]))

# Loss, miner, sampler

# Set the loss function
loss = losses.TripletMarginLoss(margin=margin)

# Set the mining function
miner = miners.MultiSimilarityMiner(epsilon=margin)

# Set the dataloader sampler
sampler = samplers.MPerClassSampler(
    train_dataset.targets, m=m_per_class, length_before_new_iter=len(train_dataset)
)

# Package the above stuff into dictionaries.
models = {"trunk": trunk, "embedder": embedder}
optimizers = {
    "trunk_optimizer": trunk_optimizer,
    "embedder_optimizer": embedder_optimizer,
}
loss_funcs = {"metric_loss": loss}
mining_funcs = {"tuple_miner": miner}

# HOOKS

record_keeper, _, _ = logging_presets.get_record_keeper(
    "lj_logs", "lj_tensorboard"
)
hooks = logging_presets.get_hook_container(record_keeper)
dataset_dict = {"val": val_dataset}
model_folder = "lj_saved_models"

def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
    logging.info(
        "UMAP plot for the {} split and label set {}".format(split_name, keyname)
    )
    label_set = np.unique(labels)
    num_classes = len(label_set)
    fig = plt.figure(figsize=(20, 15))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
    plt.show()


# Create the tester
tester = testers.GlobalEmbeddingSpaceTester(
    end_of_testing_hook=hooks.end_of_testing_hook,
    # visualizer=umap.UMAP(),
    # visualizer_hook=visualizer_hook,
    dataloader_num_workers=1,
    batch_size=eval_batch_size,
    data_device=eval_device,
    accuracy_calculator=AccuracyCalculator(k=eval_k, device=eval_device),
)

end_of_epoch_hook = hooks.end_of_epoch_hook(
    tester, dataset_dict, model_folder, test_interval=1, patience=patience
)

# TRAINER

trainer = trainers.MetricLossOnly(
    models,
    optimizers,
    batch_size,
    loss_funcs,
    mining_funcs,
    train_dataset,
    sampler=sampler,
    dataloader_num_workers=2,
    end_of_iteration_hook=hooks.end_of_iteration_hook,
    end_of_epoch_hook=end_of_epoch_hook,
)

trainer.train(num_epochs=num_epochs)
