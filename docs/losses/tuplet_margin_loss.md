# TupletMarginLoss

[Deep Metric Learning with Tuplet Margin Loss](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Deep_Metric_Learning_With_Tuplet_Margin_Loss_ICCV_2019_paper.pdf){target=_blank}

```python
losses.TupletMarginLoss(margin=5.73, scale=64, **options)
```

**Equation**:

$$
\mathcal{L}_{tuplet} = \log\left(1 + \sum_{i=1}^{k-1} e^{s\left(\cos\theta_{an_i} - \cos\left(\theta_{ap} - \beta\right)\right)}\right)
$$

Here $\theta_{ap}$ and $\theta_{an_i}$ are the angles from the anchor to its positive and to its $i$-th negative. The implementation sums over all $k - 1$ negatives of the anchor in the batch.

**Implementation notes**: Each positive pair contributes one loss term, averaged by the default reducer. Cosines are clamped just inside $[-1, 1]$ before $\arccos$, so identical anchor-positive pairs, such as duplicates from sampling with replacement, have finite values and gradients.

**Parameters**:

- **margin**: The angular margin (in degrees) applied to positive pairs. This is beta in the above equation. The paper uses a value of 5.73 degrees (0.1 radians).
- **scale**: This is `s` in the above equation.

The paper combines this loss with [IntraPairVarianceLoss](intra_pair_variance_loss.md#intrapairvarianceloss). You can accomplish this by using [MultipleLosses](multiple_losses.md#multiplelosses):

```python
complete_loss = losses.MultipleLosses(
    {"tuplet": losses.TupletMarginLoss(), "variance": losses.IntraPairVarianceLoss()},
    weights={"tuplet": 1, "variance": 0.5},
)
```

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

  - This is the only compatible distance.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per positive pair in the batch. Reduction type is `"pos_pair"`.
