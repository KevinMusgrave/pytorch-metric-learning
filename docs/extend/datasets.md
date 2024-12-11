# How to write custom datasets

1. Subclass the ```datasets.base_dataset.BaseDataset``` class
2. Add implementations for abstract methods from the base class:
    - ```download_and_remove()```
    - ```generate_split()```


```python
from pytorch_metric_learning.datasets.base_dataset import BaseDataset

class MyDataset(BaseDataset):

    def __init__(self, my_parameter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.my_parameter = self.my_parameter

    def download_and_remove(self):
        # Downloads the dataset files needed
        #
        # If you're using a dataset that you've already downloaded elsewhere,
        # just use an empty implementation
        pass

    def generate_split(self):
        # Creates a list of image paths, and saves them into self.paths
        # Creates a list of labels for the images, and saves them into self.labels
        #
        # The default training splits that need to be covered are `train`, `test`, and `train+test`
        # If you need a different split setup, override `get_available_splits(self)` to return 
        # the split names you want
        pass

```
