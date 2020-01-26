from pytorch_metric_learning import losses, miners, regularizers, samplers, trainers
import numpy as np
from torchvision import datasets, models, transforms
import torch
import logging
from utils_for_examples import MLP, Identity, ListOfModels, get_record_keeper
logging.getLogger().setLevel(logging.INFO)

import pytorch_metric_learning
logging.info("VERSION %s"%pytorch_metric_learning.__version__)


##############################
########## Training ##########
##############################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In this example, we'll take multiple trunks and embedders,
# and encapsulate them into a single trunk and embedder model.
# Note that CascadedEmbeddings does not necessarily require a complicated setup like this.
# CascadedEmbeddings just assumes that the output of your embedder
# should be partitioned into different sections, as specified by the init argument
# "embedding_sizes".
trunk1 = models.shufflenet_v2_x0_5(pretrained=True)
trunk2 = models.shufflenet_v2_x1_0(pretrained=True)
trunk3 = models.resnet18(pretrained=True)
all_trunks = [trunk1, trunk2, trunk3]
trunk_output_sizes = []

for T in all_trunks:
    trunk_output_sizes.append(T.fc.in_features)
    T.fc = Identity()

trunk = ListOfModels(all_trunks)
trunk = torch.nn.DataParallel(trunk.to(device))

# Set the embedders. Each embedder takes a corresponding trunk model output, and outputs 64-dim embeddings.
all_embedders = []
for s in trunk_output_sizes:
    all_embedders.append(MLP([s, 64]))

# The output of embedder will be of size 64*3.
embedder = ListOfModels(all_embedders, input_sizes=trunk_output_sizes)
embedder = torch.nn.DataParallel(embedder.to(device))

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

# Set the loss functions. loss0 will be applied to the first embedder, loss1 to the second embedder etc.
loss0 = losses.TripletMarginLoss(margin=0.01)
loss1 = losses.MultiSimilarityLoss(alpha=0.1, beta=40, base=0.5)
loss2 = losses.ArcFaceLoss(margin=30, num_classes=100, embedding_size=64).to(device)

# Set the mining functions. In this example we'll apply mining to the 2nd and 3rd cascaded outputs.
miner1 = miners.MultiSimilarityMiner(epsilon=0.1)
miner2 = miners.HDCMiner(filter_percentage=0.25)

# Set the dataloader sampler
sampler = samplers.MPerClassSampler(train_dataset.targets, m=4)

# Set other training parameters
batch_size = 32
num_epochs = 2
iterations_per_epoch = 100

# Package the above stuff into dictionaries.
models = {"trunk": trunk, "embedder": embedder}
optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer}
loss_funcs = {"metric_loss_0": loss0, "metric_loss_1": loss1, "metric_loss_2": loss2}
mining_funcs = {"post_gradient_miner_1": miner1, "post_gradient_miner_2": miner2}

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

trainer = trainers.CascadedEmbeddings(models=models,
                                    optimizers=optimizers,
                                    batch_size=batch_size,
                                    loss_funcs=loss_funcs,
                                    mining_funcs=mining_funcs,
                                    iterations_per_epoch=iterations_per_epoch,
                                    dataset=train_dataset,
                                    sampler=sampler,
                                    record_keeper=record_keeper,
                                    end_of_epoch_hook=end_of_epoch_hook,
                                    embedding_sizes=[64, 64, 64])

trainer.train(num_epochs=num_epochs)
