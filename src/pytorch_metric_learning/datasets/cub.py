import os
import tarfile

from ..datasets.base_dataset import BaseDataset
from ..utils.common_functions import _urlretrieve


class CUB(BaseDataset):

    DOWNLOAD_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"

    def generate_split(self):
        dir_name = CUB.DOWNLOAD_URL.split("/")[-1].replace(".tgz", "")

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

                    if class_idx not in classes:
                        continue

                    img_idx2, img_path = l2.split()
                    img_idx2 = int(img_idx2)

                    # If the image ids correspond it's a match
                    if img_idx1 == img_idx2:
                        self.paths.append(
                            os.path.join(self.root, dir_name, "images", img_path)
                        )
                        self.labels.append(class_idx)

    def download_and_remove(self):
        os.makedirs(self.root, exist_ok=True)
        download_folder_path = os.path.join(self.root, CUB.DOWNLOAD_URL.split("/")[-1])
        _urlretrieve(url=CUB.DOWNLOAD_URL, filename=download_folder_path)
        with tarfile.open(download_folder_path, "r:gz") as tar:
            tar.extractall(self.root)
        os.remove(download_folder_path)
