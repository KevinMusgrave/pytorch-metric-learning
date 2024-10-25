# How to write custom datasets

1. Subclass the ```datasets.base_dataset.BaseDatset``` class
2. Add implementations for abstract static methods from the base class:
    - ```download_and_remove()```
    - ```get_available_splits()```


```python
from pytorch_metric_learning.datasets.base_dataset import BaseDataset

class MyDataset(BaseDataset):

    def __init__(self, my_parameter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.my_parameter = self.parameter

    @staticmethod
    def download_and_remove():
        # Downloads the dataset files needed
        #
        # If you're using a dataset that you've already downloaded elsewhere,
        # just use an empty implementation
        pass

    @staticmethod
    def get_available_splits():
        # Returns the string names of the available splits
        return ["my_split1", "my_split2"]

```
