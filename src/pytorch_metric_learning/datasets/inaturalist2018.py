import json
import os
import tarfile
import zipfile

from ..datasets.base_dataset import BaseDataset
from ..utils.common_functions import _urlretrieve


class INaturalist2018(BaseDataset):

    IMG_DOWNLOAD_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz"
    TRAIN_ANN_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train2018.json.tar.gz"
    VAL_ANN_URL = (
        "https://ml-inat-competition-datasets.s3.amazonaws.com/2018/val2018.json.tar.gz"
    )

    # Link from https://github.com/Andrew-Brown1/Smooth_AP?tab=readme-ov-file#data
    SPLITS_URL = "https://drive.google.com/uc?id=1sXfkBTFDrRU3__-NUs1qBP3sf_0uMB98"

    def generate_split(self):
        with open(os.path.join(self.root, "train2018.json"), "r") as train_f:
            train_json = json.load(train_f)

        with open(os.path.join(self.root, "val2018.json"), "r") as val_f:
            val_json = json.load(val_f)

        val_imgs, val_anns = val_json["images"], val_json["annotations"]
        train_imgs, train_anns = train_json["images"], train_json["annotations"]

        imgs, anns = val_imgs + train_imgs, val_anns + train_anns

        path2id = {x["file_name"]: x["id"] for x in imgs}
        id2label = {x["image_id"]: x["category_id"] for x in anns}

        if self.split in ["train", "test"]:
            paths = self._load_split_txt(self.split)
            ids = [path2id[p] for p in paths]
            labels = [id2label[i] for i in ids]

        elif self.split == "train+test":
            train_paths = self._load_split_txt("train")
            train_ids = [path2id[p] for p in train_paths]
            train_labels = [id2label[i] for i in train_ids]

            test_paths = self._load_split_txt("test")
            test_ids = [path2id[p] for p in test_paths]
            test_labels = [id2label[i] for i in test_ids]

            paths = train_paths + test_paths
            labels = train_labels + test_labels

        self.paths = [os.path.join(self.root, p) for p in paths]
        self.labels = labels

    def _load_split_txt(self, split):
        paths = []
        with open(
            os.path.join(
                self.root, "Inat_dataset_splits", f"Inaturalist_{split}_set1.txt"
            )
        ) as f:
            for l in f:
                paths.append(l.strip())
        return paths

    def download_and_remove(self):
        download_folder_path = os.path.join(
            self.root, INaturalist2018.IMG_DOWNLOAD_URL.split("/")[-1]
        )
        _urlretrieve(
            url=INaturalist2018.IMG_DOWNLOAD_URL, filename=download_folder_path
        )
        with tarfile.open(download_folder_path, "r:gz") as tar:
            tar.extractall(self.root)
        os.remove(download_folder_path)

        download_folder_path = os.path.join(
            self.root, INaturalist2018.TRAIN_ANN_URL.split("/")[-1]
        )
        _urlretrieve(url=INaturalist2018.TRAIN_ANN_URL, filename=download_folder_path)
        with tarfile.open(download_folder_path, "r:gz") as tar:
            tar.extractall(self.root)
        os.remove(download_folder_path)

        download_folder_path = os.path.join(
            self.root, INaturalist2018.VAL_ANN_URL.split("/")[-1]
        )
        _urlretrieve(url=INaturalist2018.VAL_ANN_URL, filename=download_folder_path)
        with tarfile.open(download_folder_path, "r:gz") as tar:
            tar.extractall(self.root)
        os.remove(download_folder_path)

        download_folder_path = os.path.join(
            self.root, INaturalist2018.SPLITS_URL.split("/")[-1]
        )
        _urlretrieve(url=INaturalist2018.SPLITS_URL, filename=download_folder_path)
        with zipfile.ZipFile(download_folder_path, "r") as zip_ref:
            zip_ref.extractall(self.root)
        os.remove(download_folder_path)
