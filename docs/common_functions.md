# Common Functions

## TorchInitWrapper
A simpler wrapper to convert the torch weight initialization functions into class form, which can then be applied within loss functions. 

Example usage:
```python
from pytorch_metric_learning.utils import common_functions as c_f
import torch

# use kaiming_uniform, with a=1 and mode='fan_out'
weight_init_func = c_f.TorchInitWrapper(torch.nn.kaiming_uniform_, a=1, mode='fan_out')
loss_func = SomeClassificationLoss(..., weight_init_func=weight_init_func)
```