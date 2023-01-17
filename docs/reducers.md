# Reducers
Reducers specify how to go from many loss values to a single loss value. For example, the [ContrastiveLoss](losses.md#contrastiveloss) computes a loss for every positive and negative pair in a batch. A reducer will take all these per-pair losses, and reduce them to a single value. Here's where reducers fit in this library's flow of filters and computations:

```Your Data --> Sampler --> Miner --> Loss --> Reducer --> Final loss value```

Reducers are passed into loss functions like this:
```python
from pytorch_metric_learning import losses, reducers
reducer = reducers.SomeReducer()
loss_func = losses.SomeLoss(reducer=reducer)
loss = loss_func(embeddings, labels) # in your training for-loop
```
Internally, the loss function creates a dictionary that contains the losses and other information. The reducer takes this dictionary, performs the reduction, and returns a single value on which ```.backward()``` can be called. Most reducers are written such that they can be passed into any loss function.


## AvgNonZeroReducer
This computes the average loss, using only the losses that are greater than 0. For example, if the losses are ```[0, 2, 0, 3]```, then this reducer will return ```2.5```.
```python
reducers.AvgNonZeroReducer(**kwargs)
```
This class is equivalent to using ```ThresholdReducer(low=0)```. See [ThresholdReducer](reducers.md#thresholdreducer).

## BaseReducer
All reducers extend this class.
```python
reducers.BaseReducer(collect_stats=False)
```

**Parameters**:

* **collect_stats**: If True, will collect various statistics that may be useful to analyze during experiments. If False, these computations will be skipped. Want to make ```True``` the default? Set the global [COLLECT_STATS](common_functions.md#collect_stats) flag.


## ClassWeightedReducer
This multiplies each loss by a class weight, and then takes the average.
```python
reducers.ClassWeightedReducer(weights, **kwargs)
```

**Parameters**:

* **weights**: A tensor of weights, where ```weights[i]``` is the weight for the ith class.


## DivisorReducer
This divides each loss by a custom value specified inside the loss function. This is useful if you want to hardcode a reduction behavior in your loss function (i.e. by using DivisorReducer), while still having the option to use other reducers.
```python
reducers.DivisorReducer(**kwargs)
```
To use this reducer, the loss function must include ```divisor``` in its loss dictionary. For example, the [ProxyAnchorLoss](losses.md#proxyanchorloss) uses ```DivisorReducer``` by default, and returns the following dictionary:

```python
loss_dict = {
    "pos_loss": {
        "losses": pos_term.squeeze(0),
        "indices": loss_indices,
        "reduction_type": "element",
        "divisor": len(with_pos_proxies),
    },
    "neg_loss": {
        "losses": neg_term.squeeze(0),
        "indices": loss_indices,
        "reduction_type": "element",
        "divisor": self.num_classes,
    },
}
```

## DoNothingReducer
This returns its input. In other words, no reduction is performed. The output will be the loss dictionary that is passed into it.
```python
reducers.DoNothingReducer(**kwargs)
```

## MeanReducer
This will return the average of the losses.
```python
reducers.MeanReducer(**kwargs)
```

## MultipleReducers
This wraps multiple reducers. Each reducer is applied to a different sub-loss, as specified in the host loss function. Then the reducer outputs are summed to obtain the final loss.
```python
reducers.MultipleReducers(reducers, default_reducer=None, **kwargs)
```

**Parameters**:

* **reducers**: A dictionary mapping from strings to reducers. The strings must match sub-loss names of the host loss function.
* **default_reducer**: This reducer will be used for any sub-losses that are not included in the keys of ```reducers```. If None, then MeanReducer() will be the default.

**Example usage**:

The [ContrastiveLoss](losses.md#contrastiveloss) has two sub-losses: ```pos_loss``` for the positive pairs, and ```neg_loss``` for the negative pairs. In this example, a [ThresholdReducer](reducers.md#thresholdreducer) is used for the ```pos_loss``` and a [MeanReducer](reducers.md#meanreducer) is used for the ```neg_loss```.
```python
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.reducers import MultipleReducers, ThresholdReducer, MeanReducer
reducer_dict = {"pos_loss": ThresholdReducer(0.1), "neg_loss": MeanReducer()}
reducer = MultipleReducers(reducer_dict)
loss_func = ContrastiveLoss(reducer=reducer)
```

## PerAnchorReducer
This converts unreduced pairs to unreduced elements. For example, [NTXentLoss](losses.md#ntxentloss) returns losses per positive pair. If you used PerAnchorReducer with NTXentLoss, then the losses per pair would first be converted to losses per batch element, before being passed to the inner reducer.
```python
def aggregation_func(x, num_per_row):
    zero_denom = num_per_row == 0
    x = torch.sum(x, dim=1) / num_per_row
    x[zero_denom] = 0
    return x

reducers.PerAnchorReducer(reducer=None, 
							aggregation_func=aggregation_func, 
							**kwargs):
```

**Parameters**:

* **reducer**: The reducer that will be fed per-element losses. The default is [MeanReducer](#meanreducer)
* **aggregation_func**: A function that takes in ```(x, num_per_row)``` and returns a loss per row of ```x```. The default is the ```aggregation_func``` defined in the code snippet above. It returns the mean per row.
   	* ```x``` is an NxN array of pairwise losses, where N is the batch size.
   	* ```num_per_row``` is a size N array which indicates how many non-zero losses there are per-row of ```x```.


## SumReducer
This will return the sum of the losses.
```python
reducers.SumReducer(**kwargs)
```


## ThresholdReducer
This computes the average loss, using only the losses that fall within a specified range.

```python
reducers.ThresholdReducer(low=None, high=None **kwargs)
```

At least one of ```low``` or ```high``` must be specified.

**Parameters**:

* **low**: Losses less than this value will be ignored.
* **high**: Losses greater than this value will be ignored.

**Examples**:

- ```ThresholdReducer(low=6)```: the filter is ```losses > 6```
    - If the losses are ```[3, 7, 1, 13, 5]```, then this reducer will return ```(7+13)/2 = 10```.

- ```ThresholdReducer(high=6)```: the filter is ```losses < 6```
    - If the losses are ```[3, 7, 1, 13, 5]```, then this reducer will return ```(1+3+5)/3 = 3```.

- ```ThresholdReducer(low=6, high=12)```: the filter is ```6 < losses < 12```
    - If the losses are ```[3, 7, 1, 13, 5]```, then this reducer will return ```(7)/1 = 7```.
