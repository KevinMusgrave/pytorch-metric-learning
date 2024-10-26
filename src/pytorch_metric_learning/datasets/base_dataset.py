from PIL import Image
from torch.utils.data import Dataset
import os
from abc import ABC, abstractmethod

class BaseDataset(ABC, Dataset):

    def __init__(self, root, split="train+test", transform=None, target_transform=None, download=False):
        self.root = root

        if download:
            if not os.path.isdir(self.root):
                os.makedirs(self.root, exist_ok=False)
                self.download_and_remove()
            elif os.listdir(self.root) == []:
                self.download_and_remove()
            else:
                raise ValueError("The given directory exists and is not empty.")
        else:
            # The given directory does not exist so the user should be aware of downloading it
            # Otherwise proceed as usual
            if not os.path.isdir(self.root):
                raise ValueError("The given path does not exist. "
                    "You should probably initialize the dataset with download=True."
                )

        self.transform = transform
        self.target_transform = target_transform

        if split not in self.get_available_splits():
            raise ValueError(f"Supported splits are: {', '.join(self.get_available_splits())}")
        
        self.split = split

        self.generate_split()

    @abstractmethod
    def generate_split():
        raise NotImplementedError
    
    @abstractmethod
    def download_and_remove():
        raise NotImplementedError

    def get_available_splits(self):
        return ["train", "test", "train+test"]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label)