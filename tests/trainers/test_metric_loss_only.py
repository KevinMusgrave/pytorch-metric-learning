import tempfile
import unittest
from pathlib import Path

import torch

from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.samplers import MPerClassSampler
from pytorch_metric_learning.testers import GlobalEmbeddingSpaceTester
from pytorch_metric_learning.trainers import MetricLossOnly
from pytorch_metric_learning.utils import accuracy_calculator, logging_presets
from pytorch_metric_learning.utils import common_functions as c_f

from .. import TEST_DEVICE, TEST_DTYPES


class TestMetricLossOnly(unittest.TestCase):
    def make_datasets(self):
        generator = torch.Generator().manual_seed(23)
        labels = torch.arange(8).repeat_interleave(8)
        prototypes = torch.randn(8, 16, generator=generator)
        return {
            split: torch.utils.data.TensorDataset(
                prototypes[labels] + 0.1 * torch.randn(64, 16, generator=generator),
                labels,
            )
            for split in ("train", "val")
        }

    def assert_logging(self, hooks, tester, split_suffix):
        record_keeper = hooks.record_keeper
        for record_name in (
            "metric_loss_NTXentLoss",
            "metric_loss_NTXentLoss__modules_distance_CosineSimilarity",
            "metric_loss_NTXentLoss__modules_reducer_AvgNonZeroReducer",
        ):
            self.assertTrue(record_keeper.table_exists(record_name))
            self.assertEqual(
                len(record_keeper.query(f"SELECT * FROM {record_name}")), 6
            )

        best_epoch, best_accuracy = hooks.get_best_epoch_and_accuracy(tester, "val")
        self.assertIn(best_epoch, (1, 2))
        self.assertGreaterEqual(best_accuracy, 0)
        self.assertLessEqual(best_accuracy, 1)
        accuracies, metric_key = hooks.get_accuracies_of_best_epoch(tester, "val")
        self.assertEqual(metric_key, "precision_at_1_level0")
        self.assertEqual(
            c_f.sqlite_obj_to_dict(accuracies)[metric_key][0], best_accuracy
        )
        best = hooks.get_accuracies_of_epoch(tester, "val", best_epoch)
        self.assertEqual(c_f.sqlite_obj_to_dict(best)[metric_key][0], best_accuracy)
        history = hooks.get_accuracy_history(tester, "val")
        self.assertEqual(history["epoch"], [1, 2])
        self.assertEqual(
            history[metric_key][history["epoch"].index(best_epoch)], best_accuracy
        )
        self.assertEqual(
            hooks.get_curr_primary_metric(tester, "val"), history[metric_key][-1]
        )
        losses = hooks.get_loss_history()["metric_loss"]
        self.assertEqual(len(losses), 6)
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))
        base = "accuracies_normalized_GlobalEmbeddingSpaceTester_level_0"
        self.assertEqual(hooks.base_record_group_name(tester), base)
        self.assertEqual(
            hooks.record_group_name(tester, "val"), f"{base}_VAL_vs_{split_suffix}"
        )

    def train_and_check(self, root, dtype, splits_to_eval):
        dataset_dict = self.make_datasets()
        linear = torch.nn.Linear(16, 16)
        batchnorm = torch.nn.BatchNorm1d(16)
        model = torch.nn.Sequential(
            linear,
            batchnorm,
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
        ).to(device=TEST_DEVICE, dtype=dtype)
        initial_weight = linear.weight.detach().clone()
        assert batchnorm.running_mean is not None
        initial_running_mean = batchnorm.running_mean.detach().clone()
        record_keeper, _, tensorboard_writer = logging_presets.get_record_keeper(
            str(root / "logs"), str(root / "tensorboard")
        )
        assert tensorboard_writer is not None
        try:
            hooks = logging_presets.get_hook_container(
                record_keeper, primary_metric="precision_at_1", log_freq=2
            )
            tester = GlobalEmbeddingSpaceTester(
                end_of_testing_hook=hooks.end_of_testing_hook,
                accuracy_calculator=accuracy_calculator.AccuracyCalculator(
                    include=("precision_at_1", "AMI"), k=1
                ),
                data_device=TEST_DEVICE,
                dtype=dtype,
                dataloader_num_workers=0,
            )
            model_folder = str(root / "models")
            trainer = MetricLossOnly(
                models={"trunk": model},
                optimizers={
                    "trunk_optimizer": torch.optim.Adam(model.parameters(), lr=0.01)
                },
                batch_size=16,
                loss_funcs={"metric_loss": NTXentLoss(reducer=AvgNonZeroReducer())},
                dataset=dataset_dict["train"],
                sampler=MPerClassSampler(
                    dataset_dict["train"].tensors[1],
                    m=4,
                    batch_size=16,
                    length_before_new_iter=64,
                ),
                data_device=TEST_DEVICE,
                dtype=dtype,
                dataloader_num_workers=0,
                iterations_per_epoch=6,
                freeze_trunk_batchnorm=True,
                end_of_iteration_hook=hooks.end_of_iteration_hook,
                end_of_epoch_hook=hooks.end_of_epoch_hook(
                    tester,
                    dataset_dict,
                    model_folder,
                    test_interval=1,
                    patience=1,
                    splits_to_eval=splits_to_eval,
                ),
            )
            trainer.train(num_epochs=2)
            self.assertFalse(torch.equal(initial_weight, linear.weight))
            torch.testing.assert_close(initial_running_mean, batchnorm.running_mean)
            self.assert_logging(
                hooks, tester, "self" if splits_to_eval is None else "TRAIN_and_VAL"
            )
            checkpoint = torch.load(
                root / "models" / "trunk_2.pth",
                map_location=TEST_DEVICE,
                weights_only=True,
            )
            for name, value in model.state_dict().items():
                torch.testing.assert_close(checkpoint[name], value)
            with torch.no_grad():
                linear.weight.zero_()
            self.assertEqual(
                hooks.load_latest_saved_models(
                    trainer, model_folder, device=TEST_DEVICE
                ),
                3,
            )
            for name, value in model.state_dict().items():
                torch.testing.assert_close(checkpoint[name], value)
        finally:
            tensorboard_writer.close()

    def test_metric_loss_only(self):
        for dtype in TEST_DTYPES:
            for splits_to_eval in (
                None,
                [("train", ["train", "val"]), ("val", ["train", "val"])],
            ):
                with (
                    self.subTest(dtype=dtype, splits=splits_to_eval),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    self.train_and_check(Path(directory), dtype, splits_to_eval)
