from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import torch

class TwoStreamDataset(Dataset):
    """Two Stream dataset."""

    def __init__(self, root, posneg_transform=None, anchor_transform=None):
        """
        Args:
            root (string): path containing directory achors and posnegs
            posneg_transform (callable, optional): Optional transform to be applied on positve/negatives
            anchor_transform (callable, optional): Optional transform to be applied on anchors
        """
        self.root = root
        self.posneg_transform = posneg_transform
        self.anchor_transform = anchor_transform
        self.anchor_dataset = datasets.ImageFolder(root=root+"/anchors", transform=anchor_transform)
        self.posneg_dataset = datasets.ImageFolder(root=root+"/posnegs", transform=posneg_transform)
        self.classes = self.anchor_dataset.classes

    def __len__(self):
        return len(self.anchor_dataset.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        (anchor_img, classidx) = self.anchor_dataset.__getitem__(idx)
        (posneg_img, _) = self.posneg_dataset.__getitem__(classidx)
        return anchor_img, posneg_img, classidx