# Datasets

Datasets classes give you a way to automatically download a dataset and transform it into a PyTorch dataset.

## BaseDataset

All dataset classes extend this class and therefore inherit its ```__init__``` parameters.

```python
datasets.base_dataset.BaseDataset(
    root, 
    split="train+test", 
    transform=None, 
    target_transform=None, 
    download=False
)
```

**Parameters**:

* **root**: The path where the dataset files are saved.
* **split**: A string that determines which split of the dataset is loaded. 
* **transform**: A `torchvision.transforms` object which will be used on the input images. 
* **target_transform**: A `torchvision.transforms` object which will be used on the labels. 
* **download**: Whether to download the dataset or not. Setting this as False, but not having the dataset on the disk will raise a ValueError.


**Required Implementations**:
```python
    @staticmethod
    @abstractmethod
    def download_and_remove():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_available_splits():
        raise NotImplementedError
```

## CUB-200-2011

```python
datasets.cub.CUB(*args, **kwargs)
```

**Defined splits**: 

- `train` - Consists of examples from classes 1 to 100.
- `test` - Consists of examples from classes 101 to 200.
- `train+test` - Consists of examples from all classes.

**Loading different dataset splits**
```python
train_dataset = CUB(root="data", 
    split="train", 
    transform=None, 
    target_transform=None, 
    download=True
)
# No need to download the dataset after it is already downladed
test_dataset = CUB(root="data", 
    split="test", 
    transform=None, 
    target_transform=None, 
    download=False
)
train_and_test_dataset = CUB(root="data", 
    split="train+test", 
    transform=None, 
    target_transform=None, 
    download=False
) 
```

## Cars196

```python
datasets.cars196.Cars196(*args, **kwargs)
```

**Defined splits**: 

- `train` - Consists of examples from classes 1 to 99.
- `test` - Consists of examples from classes 99 to 197.
- `train+test` - Consists of examples from all classes.

**Loading different dataset splits**
```python
train_dataset = Cars196(root="data", 
    split="train", 
    transform=None, 
    target_transform=None, 
    download=True
)
# No need to download the dataset after it is already downladed
test_dataset = Cars196(root="data", 
    split="test", 
    transform=None, 
    target_transform=None, 
    download=False
)
train_and_test_dataset = Cars196(root="data", 
    split="train+test", 
    transform=None, 
    target_transform=None, 
    download=False
) 
```
