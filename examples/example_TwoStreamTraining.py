# The testing module requires faiss
# So if you don't have that, then this import will break 
from pytorch_metric_learning import losses, miners, samplers, trainers, testers
from TwoStreamDataset import TwoStreamDataset
import pytorch_metric_learning.utils.logging_presets as logging_presets
import numpy as np
from torchvision import datasets, models, transforms
import torch
import logging
from utils_for_examples import MLP, Identity
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
embedder = torch.nn.DataParallel(MLP([trunk_output_size, 128]).to(device))

# Set optimizers
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.00004, weight_decay=0.00005)
embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=0.00004, weight_decay=0.00005)


posneg_transform = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=227),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

anchor_transform = transforms.Compose([transforms.CenterCrop(227),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Set the datasets
dataroot = "/data/IAMAI"
train_dataset = TwoStreamDataset(root=dataroot+"/train/", anchor_transform=anchor_transform, posneg_transform=posneg_transform)
val_dataset = TwoStreamDataset(root=dataroot+"/val/", anchor_transform=anchor_transform, posneg_transform=posneg_transform)


# Set the loss function
loss = losses.TripletMarginLoss(margin=0.2)

# Set the mining function
miner = miners.TripletMarginMiner(margin=0.2)

# Set the dataloader sampler
sampler = samplers.MPerClassSampler(train_dataset.classes, m=1)

# Set other training parameters
batch_size = 16
num_epochs = 2
iterations_per_epoch = 10

# Package the above stuff into dictionaries.
models = {"trunk": trunk, "embedder": embedder}
optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer}
loss_funcs = {"metric_loss": loss}
mining_funcs = {"tuple_miner": miner}

record_keeper, _, _ = logging_presets.get_record_keeper("example_logs", "example_tensorboard")
hooks = logging_presets.get_hook_container(record_keeper)
dataset_dict = {"val": val_dataset}
model_folder = "example_saved_models"

# Create the tester
tester = testers.GlobalTwoStreamEmbeddingSpaceTester(end_of_testing_hook=hooks.end_of_testing_hook, dataloader_num_workers=2)
end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder)

trainer = trainers.TwoStreamTrainer(models=models,
                                optimizers=optimizers,
                                batch_size=batch_size,
                                loss_funcs=loss_funcs,
                                mining_funcs=mining_funcs,
                                iterations_per_epoch=iterations_per_epoch,
                                dataset=train_dataset,
                                sampler=sampler,
                                dataloader_num_workers=2,
                                end_of_iteration_hook=hooks.end_of_iteration_hook,
                                end_of_epoch_hook=end_of_epoch_hook
                                )

trainer.train(num_epochs=num_epochs)