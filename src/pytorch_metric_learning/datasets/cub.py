from ..datasets.base_dataset import BaseDataset
import os

class CUB(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        dir_name = self.get_download_url().split('/')[-1].replace(".tgz", "")

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
                        self.paths.append(img_path)
                        self.labels.append(class_idx)

        assert len(self.paths) == len(self.labels) == 11788

        # Normalize labels to start from 0
        self.labels = [x - min(self.labels) for x in self.labels]

    def download_and_remove(self):
        archive_name = self.get_download_url().split('/')[-1]
        os.system(f"wget -P {self.root} {self.get_download_url()}")
        os.system(f"cd {self.root} && tar -xzvf {archive_name}")
        os.system(f"rm {os.path.join(self.root, archive_name)}")

    @staticmethod
    def get_available_splits():
        return ["train", "test", "train+test"]

    @staticmethod
    def get_download_url():
        return "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"