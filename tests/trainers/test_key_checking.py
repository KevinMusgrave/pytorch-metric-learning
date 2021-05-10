import copy
import logging
import unittest

import torch
from torchvision import datasets

from pytorch_metric_learning.losses import NTXentLoss, TripletMarginLoss
from pytorch_metric_learning.trainers import (
    CascadedEmbeddings,
    DeepAdversarialMetricLearning,
    MetricLossOnly,
    TrainWithClassifier,
    TwoStreamMetricLoss,
)
from pytorch_metric_learning.utils import common_functions as c_f

logging.basicConfig()
logging.getLogger(c_f.LOGGER_NAME).setLevel(logging.INFO)


class TestMetricLossOnly(unittest.TestCase):
    def test_metric_loss_only(self):
        loss_fn = NTXentLoss()
        dataset = datasets.FakeData()
        model = torch.nn.Identity()
        batch_size = 32

        for trainer_class in [
            MetricLossOnly,
            DeepAdversarialMetricLearning,
            TrainWithClassifier,
            TwoStreamMetricLoss,
        ]:

            model_dict = {"trunk": model}
            optimizer_dict = {"trunk_optimizer": None}
            loss_fn_dict = {"metric_loss": loss_fn}

            if trainer_class is DeepAdversarialMetricLearning:
                model_dict["generator"] = model
                loss_fn_dict["synth_loss"] = loss_fn
                loss_fn_dict["g_adv_loss"] = TripletMarginLoss()

            kwargs = {
                "models": model_dict,
                "optimizers": optimizer_dict,
                "batch_size": batch_size,
                "loss_funcs": loss_fn_dict,
                "mining_funcs": {},
                "dataset": dataset,
                "freeze_these": ["trunk"],
            }

            trainer = trainer_class(**kwargs)

            for k in ["models", "mining_funcs", "loss_funcs", "freeze_these"]:
                new_kwargs = copy.deepcopy(kwargs)
                if k == "models":
                    new_kwargs[k] = {}
                if k == "mining_funcs":
                    new_kwargs[k] = {"dog": None}
                if k == "loss_funcs":
                    if trainer_class is DeepAdversarialMetricLearning:
                        new_kwargs[k] = {}
                    else:
                        continue
                if k == "freeze_these":
                    new_kwargs[k] = ["frog"]
                with self.assertRaises(AssertionError):
                    trainer = trainer_class(**new_kwargs)


if __name__ == "__main__":
    unittest.main()
