# DecoupledContrastiveLoss

[Decoupled Contrastive Learning](https://arxiv.org/abs/2110.06848) (Yeh et al.)
removes the positive pair from the denominator of the
[InfoNCE objective](ntxent_loss.md). In InfoNCE, every gradient is scaled by a
factor that shrinks when the batch is easy, which couples positives to negatives
and makes small batches learn slowly. Removing the positive term removes that factor.

```python {.signature}
losses.DecoupledContrastiveLoss(temperature=0.1, sigma=None, **options)
```

**Equation**:

$$
\mathcal{L}_{a,p} = -w_{a,p} \frac{s_{a,p}}{\tau} + \log \sum_{n} \exp\left(\frac{s_{a,n}}{\tau}\right)
$$

Here $a$ is the anchor, $p$ one of its positives, $n$ ranges over the negatives that
share anchor $a$, and $s$ is the similarity, cosine by default. This is equation 6 of
the paper with $w_{a,p} = 1$, the loss called DCL. With `sigma`, it is equation 7,
called DCLW, which weights harder positives more:

$$
w_{a,p} = 2 - \frac{\exp(s_{a,p} / \sigma)}{\mathrm{E}\left[\exp(s / \sigma)\right]}
$$

The expectation is the mean over the positive pairs in the batch, so $\mathrm{E}[w] = 1$,
and DCLW approaches DCL as $\sigma \to \infty$.

**Implementation notes**: One loss is computed for each positive pair whose anchor has
at least one negative, and the others are left out, because a logarithm of an empty sum
is undefined. An anchor's other positives are not in its sum. The weights are part of
the graph, as equation 7 is written, so they contribute to gradients. float16 and bfloat16 inputs accumulate in float32. With a distance where smaller means closer,
$s$ is the negated distance.

The loss can be negative, because no positive term bounds it from below, so reducers that
drop values at or below zero change the objective. `AvgNonZeroReducer` is unsuitable here.

**Paper settings**: The paper sums over both views and all samples, while the default
[MeanReducer](../reducers.md#meanreducer) averages, which only rescales the loss; use
[SumReducer](../reducers.md#sumreducer) to match the sum. For its two-view setting, wrap
the loss in [SelfSupervisedLoss](self_supervised_loss.md) with the default
`symmetric=True`, which uses both views as anchors and draws negatives from both, as in
the example below. The paper's temperature is 0.1 on ImageNet-1K, and it chooses
$\sigma = 0.5$ for DCLW.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.SelfSupervisedLoss(losses.DecoupledContrastiveLoss(sigma=0.5))
embeddings = torch.randn(32, 128, requires_grad=True)
ref_emb = torch.randn(32, 128, requires_grad=True)
loss = loss_func(embeddings, ref_emb)
loss.backward()
```

**Parameters**:

- **temperature**: The temperature $\tau$. It must be positive.
- **sigma**: The DCLW weighting scale $\sigma$, or `None` for DCL. It must be positive.
  [`anchor_batched_backward`](anchor_batched_backward.md) rejects a loss with `sigma`,
  because the weights depend on every positive pair.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)
