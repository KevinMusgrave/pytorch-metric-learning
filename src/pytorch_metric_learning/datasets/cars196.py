from ..datasets.base_dataset import BaseDataset
import os

class Cars196(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Training set is first 99 classes, test is other classes
        if self.split == "train":
            classes = set(range(1, 99))
        elif self.split == "test":
            classes = set(range(99, 197))
        else:
            classes = set(range(1, 197))
        
        paths_train, labels_train = self._load_csv(
            os.path.join(self.root, "anno_train.csv"), split="train"
        )
        paths_test, labels_test = self._load_csv(
            os.path.join(self.root, "anno_test.csv"), split="test"
        )
        paths = paths_train + paths_test
        labels = labels_train + labels_test

        self.paths, self.labels = [], []
        for p, l in zip(paths, labels):
            if l in classes:
                self.paths.append(p)
                self.labels.append(l)

    def _load_csv(self, path, split):
        all_paths, all_labels = [], []
        with open(path, "r") as f:
            for l in f:
                path_annos = l.split(",")
                curr_path = path_annos[0]
                curr_label = path_annos[-1]
                all_paths.append(
                    os.path.join(self.root, "car_data", "car_data", split, curr_path)
                )
                all_labels.append(int(curr_label))
        return all_paths, all_labels

    def download_and_remove(self):
        archive_name = self.get_download_url().split('/')[-1]
        os.system(f"wget -P {self.root} {self.get_download_url()}")
        os.system(f"cd {self.root} && unzip {archive_name}")
        os.system(f"rm {os.path.join(self.root, archive_name)}")

    @staticmethod
    def get_available_splits():
        return ["train", "test", "train+test"]

    @staticmethod
    def get_download_url():
        return "https://www.kaggle.com/api/v1/datasets/download/jutrera/stanford-car-dataset-by-classes-folder"