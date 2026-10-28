# IntraPairVarianceLoss

[Deep Metric Learning with Tuplet Margin Loss](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Deep_Metric_Learning_With_Tuplet_Margin_Loss_ICCV_2019_paper.pdf){target=_blank}

```python
losses.IntraPairVarianceLoss(pos_eps=0.01, neg_eps=0.01, **kwargs)
```

**Equation**:

$$
\mathcal{L}_{pos} = \mathbb{E}\left[(1 - \epsilon)\mu_{ap} - \cos\theta_{ap}\right]_+^2
$$

$$
\mathcal{L}_{neg} = \mathbb{E}\left[\cos\theta_{an} - (1 + \epsilon)\mu_{an}\right]_+^2
$$

Here $\theta_{ap}$ and $\theta_{an}$ are anchor-positive and anchor-negative angles, $\mu_{ap}$ and $\mu_{an}$ are the batch means of $\cos\theta_{ap}$ and $\cos\theta_{an}$, and $[\cdot]_+ = \max(0, \cdot)$.

**Implementation notes**: Gradients flow through the batch means $\mu_{ap}$ and $\mu_{an}$. With a distance where smaller means closer, the inequalities are reversed.

**Parameters**:

- **pos_eps**: The epsilon in the L<sub>pos</sub> equation. The paper uses 0.01.
- **neg_eps**: The epsilon in the L<sub>neg</sub> equation. The paper uses 0.01.

You should probably use this in conjunction with another loss, as described in the paper. You can accomplish this by using [MultipleLosses](multiple_losses.md#multiplelosses):

```python
main_loss = losses.TupletMarginLoss()
var_loss = losses.IntraPairVarianceLoss()
complete_loss = losses.MultipleLosses([main_loss, var_loss], weights=[1, 0.5])
```

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **pos_loss**: The loss per positive pair in the batch. Reduction type is `"pos_pair"`.
- **neg_loss**: The loss per negative pair in the batch. Reduction type is `"neg_pair"`.
