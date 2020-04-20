from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import torch

class TwoStreamDataset(Dataset):
    """Two Stream dataset."""

    def __init__(self, root, query_transform=None, anchor_transform=None):
        """
        Args:
            root (string): Directory with all the images.
            query_transform (callable, optional): Optional transform to be applied on queries
            ref_transform (callable, optional): Optional transform to be applied on references
        """
        self.root = root
        self.query_transform = query_transform
        self.anchor_transform = anchor_transform
        self.queries_dataset = datasets.ImageFolder(root=root+"/queries", transform=query_transform)
        self.anchor_dataset = datasets.ImageFolder(root=root+"/references", transform=anchor_transform)
        self.classes = self.queries_dataset.classes

    def __len__(self):
        return len(self.queries_dataset.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        (query_img, classidx) = self.queries_dataset.__getitem__(idx)
        (anchor_img, _) = self.anchor_dataset.__getitem__(classidx)
        return query_img, anchor_img, classidx