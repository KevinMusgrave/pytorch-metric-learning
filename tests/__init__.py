import os

import torch

dtypes_from_environ = os.environ.get("TEST_DTYPES", "float16,float32,float64").split(
    ","
)
device_from_environ = os.environ.get("TEST_DEVICE", "cpu")

TEST_DTYPES = [getattr(torch, x) for x in dtypes_from_environ]
TEST_DEVICE = torch.device(device_from_environ)
