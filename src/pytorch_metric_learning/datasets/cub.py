from PIL import Image
from torch.utils.data import Dataset
import os

class CUB(Dataset):

    SPLITS = ["train", "test", "train+test"]
    DOWNLOAD_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"

    def __init__(self, root, split="train+test", transform=None, target_transform=None, download=False):
        self.root = root

        if download and not os.path.isdir(self.root):
            archive_name = CUB.DOWNLOAD_URL.split('/')[-1]
            os.makedirs(self.root, exist_ok=False)
            os.system(f"wget -P {self.root} {CUB.DOWNLOAD_URL}")
            os.system(f"cd {self.root} && tar -xzvf {archive_name}")
            os.system(f"rm {os.path.join(self.root, archive_name)}")
        else:
            # The given directory does not exist so the user should be aware of downloading it
            # Otherwise proceed as usual
            if not os.path.isdir(self.root):
                raise ValueError("The given path does not exist. "
                    "You should probably initialize the dataset with download=True."
                )

        self.transform = transform
        self.target_transform = target_transform

        if split not in CUB.SPLITS:
            raise ValueError(f"Supported splits are: {', '.join(CUB.SPLITS)}")
        
        self.split = split

        dir_name = CUB.DOWNLOAD_URL.split('/')[-1].replace(".tgz", "")

        # Training split is first 100 classes, other 100 is test
        if self.split == "train":
            classes = set(range(1, 101))
        elif self.split == "test":
            classes = set(range(101, 201))
        else:
            classes = set(range(1, 201))
        
        # Find ids which correspond to the classes in the split
        self.paths, self.labels = [], []
        with open(os.path.join(self.root, dir_name, "image_class_labels.txt")) as f1:
            with open(os.path.join(self.root, dir_name, "images.txt")) as f2:
                for l1, l2 in zip(f1, f2):
                    img_idx1, class_idx = list(map(int, l1.split()))
                    img_idx2, img_path = l2.split()
                    img_idx2 = int(img_idx2)

                    # If the image ids correspond it's a match
                    if img_idx1 == img_idx2:
                        self.paths.append(img_path)
                        self.labels.append(class_idx)

        assert len(self.paths) == len(self.labels) == 11788

        # Normalize labels to start from 0
        self.labels = [x - min(self.labels) for x in self.labels]

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
