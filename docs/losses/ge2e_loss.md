# GE2ELoss

[Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467)
(Wan et al.) scores each embedding against the centroid of every class in the batch, and
is useful beyond speakers, for any data with several examples per class.

```python {.signature}
losses.GE2ELoss(variant="softmax", init_w=10.0, init_b=-5.0, **options)
```

**Equation**:

For embedding $e_{ji}$ of class $j$, with learnable $w > 0$ and $b$:

$$
S_{ji,k} =
\begin{cases}
w \cos\left(e_{ji}, c_j^{(-i)}\right) + b & \text{if } k = j \\
w \cos\left(e_{ji}, c_k\right) + b & \text{otherwise}
\end{cases}
$$

$c_k$ is the mean of class $k$, and $c_j^{(-i)}$ is the mean of class $j$ without $e_{ji}$.
With `variant="softmax"` (equation 6) and `variant="contrast"` (equation 7), the loss for
that embedding is:

$$
L_{\text{softmax}} = -S_{ji,j} + \log \sum_{k} \exp(S_{ji,k})
\qquad
L_{\text{contrast}} = 1 - \sigma(S_{ji,j}) + \max_{k \neq j} \sigma(S_{ji,k})
$$

**Implementation notes**: Embeddings are normalized before centroids are computed, as
equation 4 requires, so a centroid is a mean of unit vectors, then normalized for the
comparison. Leaving the anchor out applies only to its own class. Without it the paper
reports unstable training and trivial solutions. A prototype using one full centroid for
every class is a different objective.

One loss is computed for each embedding whose class has at least two embeddings in the
batch. Classes may have different sizes. A class with one embedding has no
$c_j^{(-i)}$, so that embedding is not scored, but its centroid still competes as a
negative. If no embedding remains, or the batch has one class, the result is a
differentiable zero.

$w$ is clamped to a small positive value during the forward pass, which enforces $w > 0$.
$b$ cancels in the softmax variant, so it receives a zero gradient there. float16 and
bfloat16 sums accumulate in float32.

The loss takes `labels` only. `indices_tuple` and `ref_emb` raise `ValueError`.

**Paper settings**: The defaults are the paper's initial $(w, b) = (10, -5)$. The paper sums
over all embeddings (equation 10), so use [SumReducer](../reducers.md#sumreducer) to match
the total. It also trains $w$ and $b$ with a gradient scale of 0.01, which an optimizer
parameter group with `lr` scaled by 0.01 reproduces, as in the example. The paper reports
that the contrast variant does better for text-dependent verification and the softmax
variant slightly better for text-independent verification.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

model = torch.nn.Linear(64, 32)
loss_func = losses.GE2ELoss()
optimizer = torch.optim.SGD(
    [
        {"params": model.parameters()},
        {"params": loss_func.parameters(), "lr": 0.0001},
    ],
    lr=0.01,
)
labels = torch.arange(32) // 4
loss = loss_func(model(torch.randn(32, 64)), labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**Parameters**:

- **variant**: `"softmax"` or `"contrast"`, typed as
  [`types.GE2EVariant`](../types.md#ge2evariant). Another value raises `ValueError`.
- **init_w**: The initial $w$. It must be positive.
- **init_b**: The initial $b$.

Pass the loss's parameters to an optimizer, or $w$ and $b$ stay at their initial values.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)
  - The distance must be a similarity. A distance with `is_inverted=False` raises
    `ValueError`.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)
