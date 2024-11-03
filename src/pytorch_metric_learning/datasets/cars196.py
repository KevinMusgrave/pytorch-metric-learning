import os
import zipfile

from ..datasets.base_dataset import BaseDataset
from ..utils.common_functions import _urlretrieve


class Cars196(BaseDataset):

    DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/jutrera/stanford-car-dataset-by-classes-folder"

    def generate_split(self):
        # Training set is first 99 classes, test is other classes
        if self.split == "train":
            classes = set(range(1, 99))
        elif self.split == "test":
            classes = set(range(99, 197))
        else:
            classes = set(range(1, 197))

        with open(os.path.join(self.root, "names.csv"), "r") as f:
            names = [x.strip() for x in f.readlines()]

        paths_train, labels_train = self._load_csv(
            os.path.join(self.root, "anno_train.csv"), names, split="train"
        )
        paths_test, labels_test = self._load_csv(
            os.path.join(self.root, "anno_test.csv"), names, split="test"
        )
        paths = paths_train + paths_test
        labels = labels_train + labels_test

        self.paths, self.labels = [], []
        for p, l in zip(paths, labels):
            if l in classes:
                self.paths.append(p)
                self.labels.append(l)

    def _load_csv(self, path, names, split):
        all_paths, all_labels = [], []
        with open(path, "r") as f:
            for l in f:
                path_annos = l.split(",")
                curr_path = path_annos[0]
                curr_label = path_annos[-1]
                all_paths.append(
                    os.path.join(
                        self.root,
                        "car_data",
                        "car_data",
                        split,
                        names[int(curr_label) - 1].replace("/", "-"),
                        curr_path,
                    )
                )
                all_labels.append(int(curr_label))
        return all_paths, all_labels

    def download_and_remove(self):
        os.makedirs(self.root, exist_ok=True)
        download_folder_path = os.path.join(
            self.root, Cars196.DOWNLOAD_URL.split("/")[-1]
        )
        _urlretrieve(url=Cars196.DOWNLOAD_URL, filename=download_folder_path)
        with zipfile.ZipFile(download_folder_path, "r") as zip_ref:
            zip_ref.extractall(self.root)
        os.remove(download_folder_path)
