import logging
import os
import shutil
import unittest

import numpy as np
import torch
from torchvision import datasets, transforms

from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_metric_learning.testers import GlobalEmbeddingSpaceTester
from pytorch_metric_learning.trainers import MetricLossOnly
from pytorch_metric_learning.utils import accuracy_calculator
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import logging_presets

from .. import TEST_DEVICE, TEST_DTYPES

logging.basicConfig()
logging.getLogger(c_f.LOGGER_NAME).setLevel(logging.INFO)


class TestMetricLossOnly(unittest.TestCase):
    def test_metric_loss_only(self):

        cifar_resnet_folder = "temp_cifar_resnet_for_pytorch_metric_learning_test"
        dataset_folder = "temp_dataset_for_pytorch_metric_learning_test"
        model_folder = "temp_saved_models_for_pytorch_metric_learning_test"
        logs_folder = "temp_logs_for_pytorch_metric_learning_test"
        tensorboard_folder = "temp_tensorboard_for_pytorch_metric_learning_test"

        os.system(
            "git clone https://github.com/akamaster/pytorch_resnet_cifar10.git {}".format(
                cifar_resnet_folder
            )
        )

        loss_fn = NTXentLoss()

        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize_transform,
            ]
        )

        eval_transform = transforms.Compose(
            [transforms.ToTensor(), normalize_transform]
        )

        assert not os.path.isdir(dataset_folder)
        assert not os.path.isdir(model_folder)
        assert not os.path.isdir(logs_folder)
        assert not os.path.isdir(tensorboard_folder)

        subset_idx = np.arange(10000)

        train_dataset = datasets.CIFAR100(
            dataset_folder, train=True, download=True, transform=train_transform
        )

        train_dataset_for_eval = datasets.CIFAR100(
            dataset_folder, train=True, download=True, transform=eval_transform
        )

        val_dataset = datasets.CIFAR100(
            dataset_folder, train=False, download=True, transform=eval_transform
        )

        train_dataset = torch.utils.data.Subset(train_dataset, subset_idx)
        train_dataset_for_eval = torch.utils.data.Subset(
            train_dataset_for_eval, subset_idx
        )
        val_dataset = torch.utils.data.Subset(val_dataset, subset_idx)

        for dtype in TEST_DTYPES:
            for splits_to_eval in [
                None,
                [("train", ["train", "val"]), ("val", ["train", "val"])],
            ]:
                from temp_cifar_resnet_for_pytorch_metric_learning_test import resnet

                model = torch.nn.DataParallel(resnet.resnet20())
                checkpoint = torch.load(
                    "{}/pretrained_models/resnet20-12fca82f.th".format(
                        cifar_resnet_folder
                    ),
                    map_location=TEST_DEVICE,
                )
                model.load_state_dict(checkpoint["state_dict"])
                model.module.linear = c_f.Identity()
                if TEST_DEVICE == torch.device("cpu"):
                    model = model.module
                model = model.to(TEST_DEVICE).type(dtype)

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=0.0002,
                    weight_decay=0.0001,
                    eps=1e-04,
                )

                batch_size = 32
                iterations_per_epoch = 10
                model_dict = {"trunk": model}
                optimizer_dict = {"trunk_optimizer": optimizer}
                loss_fn_dict = {"metric_loss": loss_fn}
                sampler = MPerClassSampler(
                    np.array(train_dataset.dataset.targets)[subset_idx],
                    m=4,
                    batch_size=32,
                    length_before_new_iter=len(train_dataset),
                )

                record_keeper, _, _ = logging_presets.get_record_keeper(
                    logs_folder, tensorboard_folder
                )
                hooks = logging_presets.get_hook_container(
                    record_keeper, primary_metric="precision_at_1"
                )
                dataset_dict = {"train": train_dataset_for_eval, "val": val_dataset}

                tester = GlobalEmbeddingSpaceTester(
                    end_of_testing_hook=hooks.end_of_testing_hook,
                    accuracy_calculator=accuracy_calculator.AccuracyCalculator(
                        include=("precision_at_1", "AMI"), k=1
                    ),
                    data_device=TEST_DEVICE,
                    dtype=dtype,
                    dataloader_num_workers=2,
                )

                end_of_epoch_hook = hooks.end_of_epoch_hook(
                    tester,
                    dataset_dict,
                    model_folder,
                    test_interval=1,
                    patience=1,
                    splits_to_eval=splits_to_eval,
                )

                trainer = MetricLossOnly(
                    models=model_dict,
                    optimizers=optimizer_dict,
                    batch_size=batch_size,
                    loss_funcs=loss_fn_dict,
                    mining_funcs={},
                    dataset=train_dataset,
                    sampler=sampler,
                    data_device=TEST_DEVICE,
                    dtype=dtype,
                    dataloader_num_workers=2,
                    iterations_per_epoch=iterations_per_epoch,
                    freeze_trunk_batchnorm=True,
                    end_of_iteration_hook=hooks.end_of_iteration_hook,
                    end_of_epoch_hook=end_of_epoch_hook,
                )

                num_epochs = 2
                trainer.train(num_epochs=num_epochs)
                best_epoch, best_accuracy = hooks.get_best_epoch_and_accuracy(
                    tester, "val"
                )

                accuracies, primary_metric_key = hooks.get_accuracies_of_best_epoch(
                    tester, "val"
                )
                accuracies = c_f.sqlite_obj_to_dict(accuracies)
                self.assertTrue(accuracies[primary_metric_key][0] == best_accuracy)
                self.assertTrue(primary_metric_key == "precision_at_1_level0")

                best_epoch_accuracies = hooks.get_accuracies_of_epoch(
                    tester, "val", best_epoch
                )
                best_epoch_accuracies = c_f.sqlite_obj_to_dict(best_epoch_accuracies)
                self.assertTrue(
                    best_epoch_accuracies[primary_metric_key][0] == best_accuracy
                )

                accuracy_history = hooks.get_accuracy_history(tester, "val")
                self.assertTrue(
                    accuracy_history[primary_metric_key][
                        accuracy_history["epoch"].index(best_epoch)
                    ]
                    == best_accuracy
                )

                loss_history = hooks.get_loss_history()
                if splits_to_eval is None:
                    self.assertTrue(
                        len(loss_history["metric_loss"])
                        == iterations_per_epoch * num_epochs
                    )

                curr_primary_metric = hooks.get_curr_primary_metric(tester, "val")
                self.assertTrue(
                    curr_primary_metric == accuracy_history[primary_metric_key][-1]
                )

                base_record_group_name = hooks.base_record_group_name(tester)

                self.assertTrue(
                    base_record_group_name
                    == "accuracies_normalized_GlobalEmbeddingSpaceTester_level_0"
                )

                record_group_name = hooks.record_group_name(tester, "val")

                if splits_to_eval is None:
                    self.assertTrue(
                        record_group_name
                        == "accuracies_normalized_GlobalEmbeddingSpaceTester_level_0_VAL_vs_self"
                    )
                else:
                    self.assertTrue(
                        record_group_name
                        == "accuracies_normalized_GlobalEmbeddingSpaceTester_level_0_VAL_vs_TRAIN_and_VAL"
                    )

                shutil.rmtree(model_folder)
                shutil.rmtree(logs_folder)
                shutil.rmtree(tensorboard_folder)

        shutil.rmtree(cifar_resnet_folder)
        shutil.rmtree(dataset_folder)


if __name__ == "__main__":
    unittest.main()
