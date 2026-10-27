# Common Functions

## LOGGER

This is the logger that is used everywhere in this library.

```python
from pytorch_metric_learning.utils import common_functions as c_f
c_f.LOGGER.info("Using the PML logger")
```

## LOGGER_NAME

The default logger name is `"PML"`. You can set the logging level for just this library:

```python
import logging
from pytorch_metric_learning.utils import common_functions as c_f
logging.basicConfig()
logging.getLogger(c_f.LOGGER_NAME).setLevel(logging.INFO)
```

## set_logger_name

Allows you to change `LOGGER_NAME`

```python
from pytorch_metric_learning.utils import common_functions as c_f
c_f.set_logger_name("DOGS")
c_f.LOGGER.info("Hello") # prints INFO:DOGS:Hello
```
