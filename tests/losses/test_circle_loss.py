from pytorch_metric_learning import losses, miners, samplers
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import logging
import tqdm
import numpy as np 
from sklearn.neighbors import NearestCentroid
import faiss

import pytorch_metric_learning
logging.info("VERSION %s"%pytorch_metric_learning.__version__)

class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

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

    def forward(self, x):
        return self.net(x)

# ------------------
# Training
# ------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set trunk model and replace the softmax layer with an identity function
trunk = models.resnet18(pretrained=True)
trunk_output_size = trunk.fc.in_features
trunk.fc = Identity()
trunk.to(device)

# Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
embedder = MLP([trunk_output_size, 64]).to(device)

# Set optimizers
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.00001, weight_decay=0.00005)
embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=0.00001, weight_decay=0.00005)

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
# loss = losses.TripletMarginLoss(margin=0.01)
loss = losses.CircleLoss(m=0.25, gamma=80.)

# Set the mining function
miner = miners.MultiSimilarityMiner(epsilon=0.1)

# Set the dataloader sampler
sampler = samplers.MPerClassSampler(train_dataset.targets, m=4, length_before_new_iter=3200)

# Set other training parameters
batch_size = 32
num_epochs = 20

# build loader 
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=4,
    pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=100,
    shuffle=False,
    pin_memory=True
)
val_loader_of_train_data = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=100,
    shuffle=False,
)


for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        imgs = trunk(imgs)
        emb = embedder(imgs)
        hard_pairs = miner(emb, labels)
        loss_iter = loss(emb, labels, hard_pairs)
        print("Training {}/{} - loss: {:.4f}".format(i+1, len(train_loader), loss_iter.item()), end='\r')

        trunk_optimizer.zero_grad()
        embedder_optimizer.zero_grad()
        loss_iter.backward()
        trunk_optimizer.step()
        embedder_optimizer.step()

    # evaluate
    print()
    trunk.eval()
    embedder.eval()
    with torch.no_grad():
        train_embeddings = []
        train_labels = []

        for i, (imgs, labels) in enumerate(val_loader_of_train_data):
            print('Compute centroid: {}/{}'.format(i+1, len(val_loader_of_train_data)), end='\r')
            imgs = imgs.to(device)
            imgs = trunk(imgs)
            emb = embedder(imgs)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            train_embeddings.append(emb.detach().cpu().numpy())
            train_labels.append(labels.detach().cpu().numpy())
        print()
        train_embeddings = np.concatenate(train_embeddings, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        unique_labels = np.unique(train_labels)

        # faiss database
        index = faiss.IndexFlatIP(64)
        index.add(train_embeddings)

        true_count = 0
        count = 0
        for i, (imgs, labels) in enumerate(val_loader):
            print("Evaluating: {}/{}".format(i+1, len(val_loader)), end='\r')
            imgs = imgs.to(device)
            imgs = trunk(imgs)
            emb = embedder(imgs)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            emb = emb.cpu().numpy()
            labels = labels.cpu().numpy()

            # predict via KNN with k = 5
            p_label_knn = []
            D, I = index.search(emb, k=5)
            # find predicted labels for each instance via majority vote
            n, k = I.shape
            tmp = train_labels[I.reshape(-1)].reshape(n, k)
            match_count = np.zeros((n, len(unique_labels)))
            for idx, l in enumerate(unique_labels):
                c = (tmp == l).sum(axis=1)
                match_count[:, idx] = c
            p_label_knn = unique_labels[np.argmax(match_count, axis=1)]

            # compare with gt truth
            true_count += sum(p_label_knn == labels)
            count += len(labels)
        print() 
        print("Epoch: {} -- 5NN Acc: {:.4f}".format(epoch, true_count/count))

    trunk.train()
    embedder.train()
