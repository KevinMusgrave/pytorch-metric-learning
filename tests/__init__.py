import os
import sys

import torch

sys.path.insert(0, "src")
import pytorch_metric_learning
from pytorch_metric_learning.utils import common_functions as c_f

dtypes_from_environ = os.environ.get("TEST_DTYPES", "float16,float32,float64").split(
    ","
)
device_from_environ = os.environ.get("TEST_DEVICE", "cuda")
with_collect_stats = os.environ.get("WITH_COLLECT_STATS", "false")

TEST_DTYPES = [getattr(torch, x) for x in dtypes_from_environ]
TEST_DEVICE = torch.device(device_from_environ)

assert c_f.COLLECT_STATS is False

WITH_COLLECT_STATS = True if with_collect_stats == "true" else False
c_f.COLLECT_STATS = WITH_COLLECT_STATS

print(
    f"Testing pytorch_metric_learning version {pytorch_metric_learning.__version__} with pytorch version {torch.__version__}"
)

print(
    f"TEST_DTYPES={TEST_DTYPES}, TEST_DEVICE={TEST_DEVICE}, WITH_COLLECT_STATS={WITH_COLLECT_STATS}"
)
