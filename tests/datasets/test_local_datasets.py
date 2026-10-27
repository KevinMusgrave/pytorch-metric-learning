import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pytorch_metric_learning.datasets import (
    CUB,
    Cars196,
    INaturalist2018,
    StanfordOnlineProducts,
)


class TestLocalDatasets(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write_image(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 3), (20, 40, 60)).save(path)

    def assert_splits(self, dataset_type, expected):
        for split, labels in expected.items():
            with self.subTest(dataset=dataset_type.__name__, split=split):
                dataset = dataset_type(root=str(self.root), split=split)
                self.assertEqual(dataset.labels, labels)
                self.assertEqual(len(dataset), len(labels))
                for index, label in enumerate(labels):
                    image, actual_label = dataset[index]
                    with image:
                        self.assertEqual(image.size, (4, 3))
                        self.assertEqual(actual_label, label)

    def test_cub_splits_and_transforms(self):
        def image_size(image):
            with image:
                return image.size

        base = self.root / "CUB_200_2011"
        self.write_image("CUB_200_2011/images/first.png")
        self.write_image("CUB_200_2011/images/second.png")
        (base / "image_class_labels.txt").write_text("1 1\n2 101\n")
        (base / "images.txt").write_text("1 first.png\n2 second.png\n")
        self.assert_splits(CUB, {"train": [1], "test": [101], "train+test": [1, 101]})
        dataset = CUB(
            root=str(self.root),
            split="train",
            transform=image_size,
            target_transform=lambda label: label - 1,
        )
        self.assertEqual(dataset[0], ((4, 3), 0))
        with self.assertRaisesRegex(ValueError, "Supported splits"):
            CUB(root=str(self.root), split="invalid")

    def test_cars_class_splits_merge_original_folders(self):
        (self.root / "names.csv").write_text(
            "\n".join(f"class-{i}" for i in range(1, 197))
        )
        (self.root / "anno_train.csv").write_text("first.png,0,0,4,3,1\n")
        (self.root / "anno_test.csv").write_text("second.png,0,0,4,3,99\n")
        self.write_image("car_data/car_data/train/class-1/first.png")
        self.write_image("car_data/car_data/test/class-99/second.png")
        self.assert_splits(Cars196, {"train": [1], "test": [99], "train+test": [1, 99]})

    def test_sop_metadata_splits(self):
        base = self.root / "Stanford_Online_Products"
        self.write_image("Stanford_Online_Products/chairs/first.png")
        self.write_image("Stanford_Online_Products/lamps/second.png")
        header = "image_id class_id super_class_id path\n"
        (base / "Ebay_train.txt").write_text(header + "1 7 2 chairs/first.png\n")
        (base / "Ebay_test.txt").write_text(header + "2 11 3 lamps/second.png\n")
        self.assert_splits(
            StanfordOnlineProducts, {"train": [7], "test": [11], "train+test": [7, 11]}
        )

    def test_inaturalist_split_files_select_image_ids(self):
        for split, image_id, category in [("train", 10, 8), ("val", 20, 13)]:
            filename = f"images/{split}.png"
            self.write_image(filename)
            metadata = {
                "images": [{"file_name": filename, "id": image_id}],
                "annotations": [{"image_id": image_id, "category_id": category}],
            }
            (self.root / f"{split}2018.json").write_text(json.dumps(metadata))
        splits = self.root / "Inat_dataset_splits"
        splits.mkdir()
        (splits / "Inaturalist_train_set1.txt").write_text("images/train.png\n")
        (splits / "Inaturalist_test_set1.txt").write_text("images/val.png\n")
        self.assert_splits(
            INaturalist2018, {"train": [8], "test": [13], "train+test": [8, 13]}
        )

    def test_missing_directory_does_not_download(self):
        missing = self.root / "missing"
        with self.assertRaisesRegex(ValueError, "does not exist"):
            CUB(root=str(missing))
        self.assertFalse(missing.exists())
