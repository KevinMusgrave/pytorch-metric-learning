# MultipleLosses

This is a simple wrapper for multiple losses. Pass in a list of already-initialized loss functions. Then, when you call forward on this object, it will return the sum of all wrapped losses.

```python
losses.MultipleLosses(losses, miners=None, weights=None)
```

**Parameters**:

- **losses**: A list or dictionary of initialized loss functions. On the forward call of MultipleLosses, each wrapped loss is computed and the weighted sum is returned.
- **miners**: Optional. A list or dictionary of mining functions. This allows you to pair mining functions with loss functions. For example, if `losses = [loss_A, loss_B]`, and `miners = [None, miner_B]` then no mining will be done for `loss_A`, but the output of `miner_B` will be passed to `loss_B`. The same logic applies if `losses = {"loss_A": loss_A, "loss_B": loss_B}` and `miners = {"loss_B": miner_B}`.
- **weights**: Optional. A list or dictionary of loss weights, which will be multiplied by the corresponding losses obtained by the loss functions. The default is to multiply each loss by 1. If `losses` is a list, then `weights` must be a list. If `losses` is a dictionary, `weights` must contain the same keys as `losses`.
