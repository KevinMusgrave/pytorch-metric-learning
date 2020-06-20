# Reducers
Reducers specify how to go from many loss values to a single loss value. For example, the [ContrastiveLoss](losses.md#contrastiveloss) computes a loss for every positive and negative pair in a batch. A reducer will take all these per-pair losses, and reduce them to a single value.

Reducers are passed into loss functions like this:
```python
from pytorch_metric_learning import losses, reducers
reducer = reducers.SomeReducer()
loss_func = losses.SomeLoss(reducer=reducer)
loss = loss_func(embeddings, labels) # in your training for-loop
```
Internally, the loss function creates a dictionary that contains the losses, associated indices, and reduction type. The reducer takes this dictionary, performs the reduction, and returns a single value on which ```.backward()``` can be called.


## AvgNonZeroReducer
This computes the average loss, using only the losses that are greater than 0. For example, if the losses are ```[0, 2, 0, 3]```, then this reducer will return ```2.5```.
```python
reducers.AvgNonZeroReducer(**kwargs)
```
This class is equivalent to using ```ThresholdReducer(threshold=0)```. See [ThresholdReducer](reducers.md#thresholdreducer).

## BaseReducer
All reducers extend this class.
```python
reducers.BaseReducer()
```

## ClassWeightedReducer
This multiplies each loss by a class weight, and then takes the average.
```python
reducers.ClassWeightedReducer(weights, **kwargs)
```

**Parameters**:

* **weights**: A tensor of weights, where ```weights[i]``` is the weight for the ith class.


## DivisorReducer
This divides each loss by a custom value specified inside the loss function. 
```python
reducers.DivisorReducer(**kwargs)
```
To use this reducer, the loss function must include ```divisor_summands``` in its loss dictionary. For example, the [ProxyAnchorLoss](losses.md#proxyanchorloss) uses ```DivisorReducer``` by default, and returns the following dictionary:

```python
{"pos_loss": {"losses": pos_term.squeeze(0), 
			"indices": loss_indices, 
			"reduction_type": "element", 
			"divisor_summands": {"num_pos_proxies": len(with_pos_proxies)}},
"neg_loss": {"losses": neg_term.squeeze(0), 
			"indices": loss_indices, "reduction_type": 
			"element", 
			"divisor_summands": {"num_classes": self.num_classes}},
"reg_loss": self.regularization_loss(self.proxies)
}
```

## DoNothingReducer
This returns its input. In other words, no reduction is performed. The output will be the loss dictionary that is passed into it.
```python
reducers.DivisorReducer(**kwargs)
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

## ThresholdReducer
This computes the average loss, using only the losses that are greater than ```threshold```. For example, if the losses are ```[3, 7, 1, 13, 5]```, and the threshold is 6, then this reducer will return ```10```.
```python
reducers.ThresholdReducer(threshold, **kwargs)
```

**Parameters**:

* **threshold**: All losses that fall below this value will be ignored.