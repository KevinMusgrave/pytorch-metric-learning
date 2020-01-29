# The testing module requires faiss
# So if you don't have that, then this import will break
from pytorch_metric_learning import losses, miners, samplers, trainers, testers
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
embedder = torch.nn.DataParallel(MLP([trunk_output_size, 64]).to(device))

# Set the generator model. The input size must be 3*trunk_output_size and the output must be trunk_output_size. 
generator = torch.nn.DataParallel(MLP([3*trunk_output_size, trunk_output_size, trunk_output_size], final_relu=True))

# Set optimizers
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.00001, weight_decay=0.00005)
embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=0.00001, weight_decay=0.00005)
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.00001, weight_decay=0.00005)

# Set the image transforms
train_transform = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=227),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

val_transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(227),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Set the datasets
train_dataset = datasets.CIFAR100(root="CIFAR100_Dataset", train=True, transform=train_transform, download=True)
val_dataset = datasets.CIFAR100(root="CIFAR100_Dataset", train=False, transform=val_transform, download=True)

# Set the loss function
metric_loss = losses.TripletMarginLoss(margin=0.01)
synth_loss = losses.AngularLoss(alpha=35)
g_adv_loss = losses.AngularLoss(alpha=35)

# Set the mining function
miner = miners.MultiSimilarityMiner(epsilon=0.1)

# Set the dataloader sampler
sampler = samplers.MPerClassSampler(train_dataset.targets, m=4)

# Set other training parameters
batch_size = 32
num_epochs = 2
iterations_per_epoch = 100

# Set up your models, optimizers, loss functions etc.
models = {"trunk": trunk, 
          "embedder": embedder,
          "generator": generator}

optimizers = {"trunk_optimizer": trunk_optimizer, 
              "embedder_optimizer": embedder_optimizer,
              "generator_optimizer": generator_optimizer}
              
loss_funcs = {"metric_loss": metric_loss,
              "synth_loss": synth_loss, 
              "g_adv_loss": g_adv_loss}

# Package the above stuff into dictionaries.
mining_funcs = {"post_gradient_miner": miner}

loss_weights = {"metric_loss": 1, 
                "synth_loss": 0.1,
                "g_adv_loss": 0.1,
                "g_hard_loss": 0.1,
                "g_reg_loss": 0.1}

record_keeper, _, _ = logging_presets.get_record_keeper("example_logs", "example_tensorboard")
hooks = logging_presets.get_hook_container(record_keeper)
dataset_dict = {"val": val_dataset}
model_folder = "example_saved_models"

# Create the tester
tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook=hooks.end_of_testing_hook)
end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder)
trainer = trainers.DeepAdversarialMetricLearning(models=models,
                                                optimizers=optimizers,
                                                batch_size=batch_size,
                                                loss_funcs=loss_funcs,
                                                mining_funcs=mining_funcs,
                                                iterations_per_epoch=iterations_per_epoch,
                                                dataset=train_dataset,
                                                sampler=sampler,
                                                end_of_iteration_hook=hooks.end_of_iteration_hook,
                                                end_of_epoch_hook=end_of_epoch_hook,
                                                metric_alone_epochs=0,
                                                g_alone_epochs=0,
                                                g_triplets_per_anchor=100)
  
trainer.train(num_epochs=num_epochs)
