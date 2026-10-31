# SelectivelyContrastiveTripletLoss

[Hard Negative Examples are Hard, but Useful](https://arxiv.org/abs/2007.12749) (Xuan
et al.) shows that the hardest triplets, whose negative is more similar to the anchor
than the positive is, pull embeddings together and collapse them. This loss keeps the
ordinary triplet objective for other triplets, and for hard ones it ignores the positive
and only pushes the negative away.

```python {.signature}
losses.SelectivelyContrastiveTripletLoss(
    temperature=0.1, hard_weight=1.0, hard_threshold=None, **options
)
```

**Equation**:

$$
\mathcal{L}_{SC}(S_{ap}, S_{an}) =
\begin{cases}
\lambda S_{an} & \text{if } S_{an} > S_{ap} \\
-\log \dfrac{\exp(S_{ap} / \tau)}{\exp(S_{ap} / \tau) + \exp(S_{an} / \tau)} & \text{otherwise}
\end{cases}
$$

$S_{ap}$ and $S_{an}$ are the anchor-positive and anchor-negative similarities, cosine
by default.

**Implementation notes**: One loss is computed for each triplet. Without an
`indices_tuple`, those are all triplets in the batch. A hard triplet sends no gradient
to its positive. The second case is computed as
$\operatorname{softplus}((S_{an} - S_{ap}) / \tau)$. The two cases are not continuous at
$S_{an} = S_{ap}$, and an exact tie takes the second case. Hard terms can be negative, so
reducers that drop values at or below zero change the objective.

**Paper settings**: This is equation 13 of the paper, with $\lambda = 1$ for every
dataset. Three details need attention:

- The paper pairs this loss with hard-negative mining. Pass
  `BatchEasyHardMiner(pos_strategy="easy", neg_strategy="hard")` as in the example,
  which is the selection in the
  [authors' code](https://github.com/littleredxh/HardNegative). The paper's figures also
  show a randomly selected positive.
- Equation 1 of the paper has no temperature, which is `temperature=1`. The authors' code
  uses 0.1, the default here.
- The authors' code also treats a triplet as hard when $S_{an} > 0.8$. The paper's
  equation does not, so the default is `hard_threshold=None`. Use `hard_threshold=0.8` to
  match the code. The code then drops exact ties and removes anchors with duplicate
  images, which this loss does not.

**Example**:

```python
import torch
from pytorch_metric_learning import losses, miners

loss_func = losses.SelectivelyContrastiveTripletLoss()
miner = miners.BatchEasyHardMiner(pos_strategy="easy", neg_strategy="hard")
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.arange(32) // 4
loss = loss_func(embeddings, labels, miner(embeddings, labels))
loss.backward()
```

**Parameters**:

- **temperature**: The temperature $\tau$ of the second case. It must be positive.
- **hard_weight**: The weight $\lambda$ of the first case.
- **hard_threshold**: If set, a triplet is also hard when $S_{an}$ is above this value.

**Default distance**:

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)
  - The distance must be a similarity. A distance with `is_inverted=False` raises
    `ValueError`, because the first case minimizes the value directly.

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)
