# Common Functions

## LOGGER
This is the logger that is used everywhere in this library.
```python
from pytorch_metric_learning.utils import common_functions as c_f
c_f.LOGGER.info("Using the PML logger")
```

## LOGGER_NAME
The default logger name is ```"PML"```. You can set the logging level for just this library:
```python
import logging
from pytorch_metric_learning.utils import common_functions as c_f
logging.basicConfig()
logging.getLogger(c_f.LOGGER_NAME).setLevel(logging.INFO)
```

## set_logger_name
Allows you to change ```LOGGER_NAME```
```python
from pytorch_metric_learning.utils import common_functions as c_f
c_f.set_logger_name("DOGS")
c_f.LOGGER.info("Hello") # prints INFO:DOGS:Hello
```

## COLLECT_STATS
Default value is ```True```. This is used by all distances, losses, miners, reducers, and regularizers. Set this to ```False``` if you want to turn off all statistics collection.
```python
from pytorch_metric_learning.utils import common_functions as c_f
c_f.COLLECT_STATS = False
```

## NUMPY_RANDOM
Default value is ```np.random```. This is used anytime a numpy random function is needed. You can set it to something else if you want
```python
import numpy as np
from pytorch_metric_learning.utils import common_functions as c_f
c_f.NUMPY_RANDOM = np.random.RandomState(42)
```


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