from pytorch_metric_learning import losses, miners, samplers, trainers
import numpy as np
from torchvision import datasets, models, transforms
import torch
import logging
from utils_for_examples import MLP, Identity, get_record_keeper
logging.getLogger().setLevel(logging.INFO)

import pytorch_metric_learning
logging.info("VERSION %s"%pytorch_metric_learning.__version__)


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

# Set the dataloader sampler
sampler = samplers.MPerClassSampler(train_dataset.targets, m=4)

# Set other training parameters
batch_size = 32
num_epochs = 3
iterations_per_epoch = 100

# Package the above stuff into dictionaries.
models = {"trunk": trunk, "embedder": embedder}
optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer}
loss_funcs = {"metric_loss": loss}
mining_funcs = {"post_gradient_miner": miner}

record_keeper = get_record_keeper()

# The testing module requires faiss
# So if you don't have these, then this import will break
from pytorch_metric_learning import testers

# Create the tester
tester = testers.GlobalEmbeddingSpaceTester(record_keeper=record_keeper)
dataset_dict = {"train": train_dataset, "val": val_dataset}

# This hook will be passed into the trainer and will be executed at the end of every epoch.
def end_of_epoch_hook(trainer):
    tester.test(dataset_dict, trainer.epoch, trainer.models["trunk"], trainer.models["embedder"])
    if trainer.record_keeper is not None:
        trainer.record_keeper.pickler_and_csver.save_records()

trainer = trainers.MetricLossOnly(models,
                                optimizers,
                                batch_size,
                                loss_funcs,
                                mining_funcs,
                                iterations_per_epoch,
                                train_dataset,
                                sampler=sampler,
                                record_keeper=record_keeper,
                                end_of_epoch_hook=end_of_epoch_hook)

trainer.train(num_epochs=num_epochs)

