import os
import sys

import torch

sys.path.insert(0, "src")
import pytorch_metric_learning

dtypes_from_environ = os.environ.get("TEST_DTYPES", "float16,float32,float64").split(
    ","
)
device_from_environ = os.environ.get("TEST_DEVICE", "cuda")

TEST_DTYPES = [getattr(torch, x) for x in dtypes_from_environ]
TEST_DEVICE = torch.device(device_from_environ)

print(
    f"Testing pytorch_metric_learning version {pytorch_metric_learning.__version__} "
    f"with pytorch version {torch.__version__}"
)

print(f"TEST_DTYPES={TEST_DTYPES}, TEST_DEVICE={TEST_DEVICE}")
