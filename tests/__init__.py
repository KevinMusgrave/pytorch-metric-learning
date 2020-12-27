import os
import sys

import torch

sys.path.insert(0, "../pytorch-metric-learning/src")
import pytorch_metric_learning

print(
    "testing pytorch_metric_learning version {} with pytorch version {}".format(
        pytorch_metric_learning.__version__, torch.__version__
    )
)

dtypes_from_environ = os.environ.get("TEST_DTYPES", "float16,float32,float64").split(
    ","
)
device_from_environ = os.environ.get("TEST_DEVICE", "cuda")

TEST_DTYPES = [getattr(torch, x) for x in dtypes_from_environ]
TEST_DEVICE = torch.device(device_from_environ)
