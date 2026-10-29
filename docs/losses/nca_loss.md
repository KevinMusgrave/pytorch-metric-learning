# NCALoss

[Neighbourhood Components Analysis](https://www.cs.toronto.edu/~hinton/absps/nca.pdf)

```python
losses.NCALoss(softmax_scale=1, **options)
```

**Equation**:

$$
L = -\frac{1}{|V|} \sum_{i \in V} w_i \log p_i,
\qquad
p_i = \sum_{j \in C_i} p_{ij},
\qquad
p_{ij} = \frac{\exp\left(-s \lVert \mathbf{x}_i - \mathbf{r}_j \rVert^2\right)}{\sum_{k \in R_i} \exp\left(-s \lVert \mathbf{x}_i - \mathbf{r}_k \rVert^2\right)}
$$

Here $\mathbf{r}_j$ are the reference embeddings, $R_i$ is the set of references compared with query $i$, $C_i \subseteq R_i$ holds the references with the same label as $i$, $s$ is `softmax_scale`, $V$ is the set of valid queries, and $w_i$ is the [miner weight](../losses.md#miner-weights) of query $i$, 1 without an `indices_tuple`. The paper maximizes $\sum_i p_i$ or $\sum_i \log p_i$ with a linear map $A$, $\mathbf{x}_i = A\mathbf{z}_i$; this loss is the negated, averaged log form.

**Implementation notes**: Without `ref_emb`, the references are the other
embeddings in the batch, so $R_i$ excludes $i$ itself, as in the paper's
$p_{ii} = 0$. With `ref_emb`, $R_i$ is every reference row. A query is valid
when it has at least one positive reference; other queries are skipped. The
loss is computed in log space as
$\log \sum_{k \in R_i} e^{\cdot} - \log \sum_{j \in C_i} e^{\cdot}$, so the loss
and its gradient stay finite when the individual $p_{ij}$ underflow. float16 and bfloat16 inputs accumulate in float32. With a
similarity distance, the exponents are $s \cdot \mathrm{sim}(\mathbf{x}_i,
\mathbf{r}_j)$. The default distance L2-normalizes the embeddings.

**Paper settings**: The paper uses squared Euclidean distances between
unnormalized embeddings and sums $\log p_i$ over the points; the default
distance normalizes the embeddings and the default reducer averages. Pass
`distance=distances.LpDistance(normalize_embeddings=False, power=2)` and
`reducer=reducers.SumReducer()` for the paper's objective.

**Example**:

```python
import torch
from pytorch_metric_learning import losses

loss_func = losses.NCALoss()
embeddings = torch.randn(32, 128, requires_grad=True)
labels = torch.randint(0, 8, (32,))
loss = loss_func(embeddings, labels)
loss.backward()
```

**Parameters**:

- **softmax_scale**: The multiplier $s$ of the distances in the softmax. The
  paper uses 1.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=2)`](../distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: One loss per valid query, indexed by the query's position in
  `embeddings`.
