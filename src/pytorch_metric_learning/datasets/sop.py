from ..datasets.base_dataset import BaseDataset
from ..utils.common_functions import _urlretrieve
import os
import zipfile

class StanfordOnlineProducts(BaseDataset):

    DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id=1TclrpQOF_ullUP99wk_gjGN8pKvtErG8&export=download&authuser=0&confirm=t"

    def generate_split(self):
        if self.split in ["train", "test"]:
            paths, labels = self._load_split_txt(self.split)
        elif self.split == "train+test":
            train_paths, train_labels = self._load_split_txt("train")
            test_paths, test_labels = self._load_split_txt("test")
            paths = train_paths + test_paths
            labels = train_labels + test_labels

        self.paths = paths
        self.labels = labels

    def _load_split_txt(self, split):
        paths, labels = [], []
        with open(os.path.join(self.root, "Stanford_Online_Products", f"Ebay_{split}.txt")) as f:
            for i, l in enumerate(f):
                if i == 0:
                    continue
                l_split = l.strip().split()
                label, path = int(l_split[1]), l_split[3] 
                paths.append(os.path.join(self.root, "Stanford_Online_Products", path))
                labels.append(label)
        return paths, labels
    
    def download_and_remove(self):
        os.makedirs(self.root, exist_ok=True)
        download_folder_path = os.path.join(self.root, StanfordOnlineProducts.DOWNLOAD_URL.split('/')[-1])
        _urlretrieve(url=StanfordOnlineProducts.DOWNLOAD_URL, filename=download_folder_path)
        with zipfile.ZipFile(download_folder_path, "r") as zip_ref:
            zip_ref.extractall(self.root)
        os.remove(download_folder_path)
    
# if __name__ == "__main__":
#     train_dataset = StanfordOnlineProducts(root="data_sop", split="train", download=True)
#     train_dataset = StanfordOnlineProducts(root="data_sop", split="test", download=True)
#     train_dataset = StanfordOnlineProducts(root="data_sop", split="train+test", download=True)

