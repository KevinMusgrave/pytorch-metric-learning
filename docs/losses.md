# Losses

All loss functions are used as follows:

```python
from pytorch_metric_learning import losses
loss_func = losses.SomeLoss()
loss = loss_func(embeddings, labels) # in your training for-loop
```

Or if you are using a loss in conjunction with a [miner](miners.md):

```python
from pytorch_metric_learning import miners
miner_func = miners.SomeMiner()
loss_func = losses.SomeLoss()
miner_output = miner_func(embeddings, labels) # in your training for-loop
loss = loss_func(embeddings, labels, miner_output)
```

For some losses, you don't need to pass in labels if you are already passing in pair/triplet indices:

```python
loss = loss_func(embeddings, indices_tuple=pairs)
# it also works with ref_emb
loss = loss_func(embeddings, indices_tuple=pairs, ref_emb=ref_emb)
```

**Losses for which you can pass in `indices_tuple` without `labels`**

- CircleLoss
- ContrastiveLoss
- IntraPairVarianceLoss
- GeneralizedLiftedStructureLoss
- LiftedStructureLoss
- MarginLoss
- MultiSimilarityLoss
- NTXentLoss
- SignalToNoiseRatioContrastiveLoss
- SupConLoss
- TripletMarginLoss
- TupletMarginLoss

You can specify how losses get reduced to a single value by using a [reducer](reducers.md):

```python
from pytorch_metric_learning import reducers
reducer = reducers.SomeReducer()
loss_func = losses.SomeLoss(reducer=reducer)
loss = loss_func(embeddings, labels) # in your training for-loop
```

For tuple losses, you can separate the source of anchors and positives/negatives:

```python
loss_func = losses.SomeLoss()
# anchors will come from embeddings
# positives/negatives will come from ref_emb
loss = loss_func(embeddings, labels, ref_emb=ref_emb, ref_labels=ref_labels)
```

For classification losses, you can get logits using the `get_logits` function:

```python
loss_func = losses.SomeClassificationLoss()
logits = loss_func.get_logits(embeddings)
```

## Regularization terms

Every loss accepts `embedding_regularizer` and `embedding_reg_weight`. Losses
with [learnable weights](#learnable-weight-arguments) also accept
`weight_regularizer` and `weight_reg_weight`. The total loss is

$$
L = \operatorname{reduce}(\text{objective terms})
+ w_e \, R_e(\text{embeddings}) + w_w \, R_w(\text{weights}),
$$

where each penalty appears only when its regularizer is supplied. The
[reducer](reducers.md) sees only the loss's own objective terms. Each
regularizer applies its own reducer, and the weighted result is added after
reduction. With `DoNothingReducer`, the returned dictionary also contains
`"embedding_reg_loss"` and `"weight_reg_loss"` entries whose `"losses"` value is
the weighted scalar penalty and whose `"reduction_type"` is
`"already_reduced"`.

```python
from pytorch_metric_learning import losses, regularizers

loss_func = losses.ContrastiveLoss(
    embedding_regularizer=regularizers.LpRegularizer(),
    embedding_reg_weight=0.1,
)
```

## AngularLoss

[Deep Metric Learning with Angular Loss](https://arxiv.org/pdf/1708.01682.pdf){target=_blank}

```python
losses.AngularLoss(alpha=40, **kwargs)
```

**Equation**:

$$
\begin{aligned}
l_{ang}(\mathcal{B}) &= \frac{1}{N} \sum_{\mathbf{x}_a \in \mathcal{B}} \left\lbrace \log \left[ 1 + \sum_{\substack{\mathbf{x}_n \in \mathcal{B} \\ y_n \neq y_a, y_p}} \exp\left(f_{a,p,n}\right) \right] \right\rbrace \\
f_{a,p,n} &= 4 \tan^2 \alpha \, (\mathbf{x}_a + \mathbf{x}_p)^T \mathbf{x}_n - 2 \left(1 + \tan^2 \alpha\right) \mathbf{x}_a^T \mathbf{x}_p
\end{aligned}
$$

Here $N$ is the batch size. Terms that depend only on $\lVert \mathbf{x} \rVert$ are dropped from $f_{a,p,n}$, as in the N-pair loss.

**Implementation notes**: Anchors, positives, and negatives are L2-normalized, matching the unit-length derivation of the equation. The loss is averaged over all ordered positive pairs rather than one positive per anchor; pairs without negatives contribute 0.

**Parameters**:

- **alpha**: The angle specified in degrees. The paper uses values between 36 and 55.

**Default distance**:

- [`LpDistance(p=2, power=1, normalize_embeddings=True)`](distances.md#lpdistance)

  - This is the only compatible distance.

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss for every `a1`, where `(a1,p)` represents every positive pair in the batch. Reduction type is `"pos_pair"`.

## ArcFaceLoss

[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.07698.pdf){target=_blank}

```python
losses.ArcFaceLoss(num_classes, embedding_size, margin=28.6, scale=64,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **kwargs)
```

**Equation**:

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s \cos\left(\theta_{y_i} + m\right)}}{e^{s \cos\left(\theta_{y_i} + m\right)} + \sum_{j=1, j \neq y_i}^{n} e^{s \cos \theta_j}}
$$

Here $N$ is the batch size, $n$ is `num_classes`, and $\theta_j$ is the angle between an embedding and the weight vector of class $j$.

**Implementation notes**: When $\theta_{y_i} > \pi - m$, the target logit is $\cos\theta_{y_i} - m \sin m$, which keeps it decreasing in $\theta$. The default margin of 28.6 degrees is 0.4992 radians; pass `margin=28.648` for the paper's 0.5. With an `indices_tuple`, each element's loss is weighted by its relative mining frequency.

**Parameters**:

- **num_classes**: The number of classes in your training dataset.
- **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set `embedding_size` to 512.
- **margin**: The angular margin penalty in degrees. In the above equation, `m = radians(margin)`. The paper uses 0.5 radians, which is 28.6 degrees.
- **scale**: This is `s` in the above equation. The paper uses 64.

**Other info**:

- Accepts the keyword-only [learnable weight arguments](#learnable-weight-arguments) `weight_regularizer`, `weight_reg_weight`, and `weight_init_func`.
- This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:

```python
loss_func = losses.ArcFaceLoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)
  - This is the only compatible distance.

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. Reduction type is `"element"`.

## BaseMetricLossFunction

All loss functions extend this class and therefore inherit its `__init__` parameters.

```python
losses.BaseMetricLossFunction(reducer = None,
							distance = None,
							embedding_regularizer = None,
							embedding_reg_weight = 1)
```

**Parameters**:

- **reducer**: A [reducer](reducers.md) object. If None, then the default reducer will be used.
- **distance**: A [distance](distances.md) object. If None, then the default distance will be used.
- **embedding_regularizer**: A [regularizer](regularizers.md) object that will be applied to embeddings. If None, then no embedding regularization will be used.
- **embedding_reg_weight**: If an embedding regularizer is used, then its loss will be multiplied by this amount before being added to the reduced loss.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Regularization**: `embedding_regularizer` and `embedding_reg_weight` are
described in [regularization terms](#regularization-terms).

**Required Implementations**:

```python
def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
    raise NotImplementedError
```

## CircleLoss

[Circle Loss: A Unified Perspective of Pair Similarity Optimization](https://arxiv.org/pdf/2002.10857.pdf){target=_blank}

```python
losses.CircleLoss(m=0.4, gamma=80, **kwargs)
```

**Equations**:

$$
\mathcal{L}_{circle} = \log \left[ 1 + \sum_{j=1}^{L} \exp\left(\gamma \alpha_n^j \left(s_n^j - \Delta_n\right)\right) \sum_{i=1}^{K} \exp\left(-\gamma \alpha_p^i \left(s_p^i - \Delta_p\right)\right) \right]
$$

For each anchor, $s_p^i$ and $s_n^j$ are its cosine similarities to its $K$ positives and $L$ negatives.

where

$$
\begin{aligned}
\alpha_p^i &= \left[O_p - s_p^i\right]_+ \\
\alpha_n^j &= \left[s_n^j - O_n\right]_+
\end{aligned}
$$

$$
\begin{aligned}
O_p &= 1 + m, & O_n &= -m, \\
\Delta_p &= 1 - m, & \Delta_n &= m
\end{aligned}
$$

**Implementation notes**: $\alpha_p$ and $\alpha_n$ are treated as constants for gradients. Anchors without a positive or a negative contribute 0; the default reducer averages the anchors whose loss is positive.

**Parameters**:

- **m**: The relaxation factor that controls the radius of the decision boundary. The paper uses 0.25 for face recognition, and 0.4 for fine-grained image retrieval (images of birds, cars, and online products).
- **gamma**: The scale factor that determines the largest scale of each similarity score. The paper uses 256 for face recognition, and 80 for fine-grained image retrieval.

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)

  - This is the only compatible distance.

**Default reducer**:

- [AvgNonZeroReducer](reducers.md#avgnonzeroreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. Reduction type is `"element"`.

## ContrastiveLoss

```python
losses.ContrastiveLoss(pos_margin=0, neg_margin=1, **kwargs):
```

**Equation**:

If using a distance metric like [LpDistance](distances.md#lpdistance), the loss is:

$$
L_{contrastive} = \left[d_p - m_{pos}\right]_+ + \left[m_{neg} - d_n\right]_+
$$

Here $d_p$ and $d_n$ are positive-pair and negative-pair distances, $m_{pos}$ and $m_{neg}$ are `pos_margin` and `neg_margin`, and $[x]_+ = \max(0, x)$.

If using a similarity metric like [CosineSimilarity](distances.md#cosinesimilarity), the loss is:

$$
L_{contrastive} = \left[m_{pos} - s_p\right]_+ + \left[s_n - m_{neg}\right]_+
$$

Here $s_p$ and $s_n$ are positive-pair and negative-pair similarities.

**Implementation notes**: The hinges are not squared. The default reducer averages the positive-pair losses greater than zero and, separately, the negative-pair losses greater than zero, then adds the two averages.

**Parameters**:

- **pos_margin**: The distance (or similarity) over (under) which positive pairs will contribute to the loss.
- **neg_margin**: The distance (or similarity) under (over) which negative pairs will contribute to the loss.

Note that the default values for `pos_margin` and `neg_margin` are suitable if you are using a non-inverted distance measure, like [LpDistance](distances.md#lpdistance). If you use an inverted distance measure like [CosineSimilarity](distances.md#cosinesimilarity), then more appropriate values would be `pos_margin = 1` and `neg_margin = 0`.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

**Default reducer**:

- [AvgNonZeroReducer](reducers.md#avgnonzeroreducer)

**Reducer input**:

- **pos_loss**: The loss per positive pair in the batch. Reduction type is `"pos_pair"`.
- **neg_loss**: The loss per negative pair in the batch. Reduction type is `"neg_pair"`.

## CosFaceLoss

[CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.09414.pdf){target=_blank}

```python
losses.CosFaceLoss(num_classes, embedding_size, margin=0.35, scale=64,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **kwargs)
```

**Equation**:

$$
L_{lmc} = \frac{1}{N} \sum_{i} -\log \frac{e^{s \left(\cos\left(\theta_{y_i,i}\right) - m\right)}}{e^{s \left(\cos\left(\theta_{y_i,i}\right) - m\right)} + \sum_{j \neq y_i} e^{s \cos\left(\theta_{j,i}\right)}}
$$

Here $N$ is the batch size and $\theta_{j,i}$ is the angle between embedding $i$ and the weight vector of class $j$.

**Parameters**:

- **num_classes**: The number of classes in your training dataset.
- **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set `embedding_size` to 512.
- **margin**: The cosine margin penalty (m in the above equation). The paper used values between 0.25 and 0.45.
- **scale**: This is `s` in the above equation. The paper uses 64.

**Other info**:

- Accepts the keyword-only [learnable weight arguments](#learnable-weight-arguments) `weight_regularizer`, `weight_reg_weight`, and `weight_init_func`.
- This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:

```python
loss_func = losses.CosFaceLoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)

  - This is the only compatible distance.

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. Reduction type is `"element"`.

## CrossBatchMemory

This wraps a loss function, and implements [Cross-Batch Memory for Embedding Learning](https://arxiv.org/pdf/1912.06798.pdf){target=_blank}. It stores embeddings from previous iterations in a queue, and uses them to form more pairs/triplets with the current iteration's embeddings.

```python
losses.CrossBatchMemory(loss, embedding_size, memory_size=1024, miner=None, *, label_dtype=torch.long)
```

**Parameters**:

- **loss**: The loss function to be wrapped. For example, you could pass in `ContrastiveLoss()`.
- **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set `embedding_size` to 512.
- **memory_size**: The size of the memory queue.
- **miner**: An optional [tuple miner](miners.md), which will be used to mine pairs/triplets from the memory queue.
- **label_dtype**: The dtype of the label queue. Incoming labels are cast to it when the cast is safe under `torch.can_cast`; otherwise the call raises `ValueError`. Use a floating dtype for non-integer labels.

The queue buffers belong to the module. Move it with `loss_fn.to(device, dtype)`
so the embedding queue matches the embeddings; outside autocast a mismatch
raises `ValueError`. Forward never moves or casts the queue.
Under autocast, the incoming embeddings are converted to the queue's dtype for
mining and loss computation. Anchor gradients flow through this conversion;
enqueued embeddings are detached, and the queue keeps its configured precision.

**Forward function**

```python
loss_fn(embeddings, labels, indices_tuple=None, enqueue_mask=None)
```

As shown above, CrossBatchMemory comes with a 4th argument in its `forward` function:

- **enqueue_mask**: A boolean tensor where `enqueue_mask[i]` is True if `embeddings[i]` should be added to the memory queue. This enables CrossBatchMemory to be used in self-supervision frameworks like [MoCo](https://arxiv.org/pdf/1911.05722.pdf). See the [memory recipe](recipes/training.md#memory-warmup-contract) for caller-controlled queue filling.

With an enqueue mask, selected rows become detached references and unselected
rows are anchors. Without a mask, all rows are enqueued and used as anchors,
with self-comparisons excluded. The number of enqueued rows must not exceed
`memory_size`. An empty enqueue preserves the existing queue; an empty batch
returns a differentiable zero. If no references have been enqueued, the loss is
also zero.

Each forward snapshots the occupied references, so accumulating multiple forward
losses before calling backward does not let later queue writes change an earlier
loss's gradients. This copies the occupied queue and costs memory proportional
to its size; queued embeddings never retain their original computation graphs.

Save the loss module's `state_dict` to resume the queue. It includes the memory
buffers, queue cursor, and filled flag, and supports loading with
`torch.load(..., weights_only=True)`. Construct the same loss and queue dimensions
before loading. Use `load_state_dict(state, assign=True)` to retain the saved
tensor dtypes, and construct optimizers after loading. Older checkpoints that
contain only the buffers lack the `queue_state` entry, which strict loading
reports as a missing key.

**Supported Loss Functions**

- [AngularLoss](losses.md#angularloss)
- [CircleLoss](losses.md#circleloss)
- [ContrastiveLoss](losses.md#contrastiveloss)
- [GeneralizedLiftedStructureLoss](losses.md#generalizedliftedstructureloss)
- [IntraPairVarianceLoss](losses.md#intrapairvarianceloss)
- [MarginLoss](losses.md#marginloss)
- [MultiSimilarityLoss](losses.md#multisimilarityloss)
- [NCALoss](losses.md#ncaloss)
- [NTXentLoss](losses.md#ntxentloss)
- [SignalToNoiseRatioContrastiveLoss](losses.md#signaltonoiseratiocontrastiveloss)
- [SupConLoss](losses.md#supconloss)
- [TripletMarginLoss](losses.md#tripletmarginloss)
- [TupletMarginLoss](losses.md#tupletmarginloss)

**Reset queue**

The queue can be cleared like this:

```python
loss_fn.reset_queue()
```

## DynamicSoftMarginLoss

[Learning Local Descriptors With a CDF-Based Dynamic Soft Margin](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Learning_Local_Descriptors_With_a_CDF-Based_Dynamic_Soft_Margin_ICCV_2019_paper.pdf)

```python
losses.DynamicSoftMarginLoss(min_val=-2.0, num_bins=10, momentum=0.01, **kwargs)
```

The running histogram is a buffer of the loss module. Move the loss with
`loss_func.to(device, dtype)`; outside autocast, a placement mismatch raises
`ValueError`.
Under autocast, bin coordinates and histogram contributions use the buffer's
dtype even when distance differences have lower precision. Updating the
histogram does not change its dtype.

**Parameters**:

- **min_val**: finite negative lower bound of the histogram. Its upper bound is
  `-min_val`. The default range `[-2, 2]` covers differences of Euclidean
  distances between normalized embeddings.
- **num_bins**: number of equally spaced bin centers including both endpoints;
  at least 2. The library default is 10; the authors' implementation uses 512.
- **momentum**: weight of the current batch's normalized histogram, between 0
  and 1. The first nonempty batch initializes the histogram regardless of this
  setting. Empty valid sets do not update it.

With labels or supplied triplets, this loss weights `d_positive - d_negative`
for each selected triplet. With neither, row `i` in `embeddings` is matched to
row `i` in `ref_emb`; all off-diagonal candidates are negatives. Each matched
pair uses the closest negative from either stream. If stream sizes differ,
extra rows provide negatives but do not create additional positive pairs.
This path does not apply the Brown-dataset outlier threshold used by the
authors' descriptor-training code.

Each difference contributes linearly to its two adjacent histogram centers.
The cumulative probability at the lower center weights the original difference.
Values outside the histogram range use the nearest endpoint. Histogram updates
and weights are detached; gradients flow through the distances. The compatible
distance is normalized, unsquared Euclidean distance.

The running histogram is saved in `state_dict`. It supports ordinary successive
training steps and multiple forwards before backward. Starting a new module
starts a new histogram. The corrected endpoint spacing and per-batch momentum
change numerical results from version 2.9.0.

The original PML implementation was contributed by **Domenico Muscillo**.
See the [authors' reference code](https://github.com/lg-zhang/dynamic-soft-margin-pytorch)
for the descriptor-training setup.

## FastAPLoss

[Deep Metric Learning to Rank](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.pdf){target=_blank}

```python
losses.FastAPLoss(num_bins=10, **kwargs)
```

**Parameters**:

- **num_bins**: The number of soft histogram bins for calculating average precision. The paper suggests using 10.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=2)`](distances.md#lpdistance)
  - The only compatible distance is `LpDistance(normalize_embeddings=True, p=2)`. However, the `power` value can be changed.

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element that has at least 1 positive in the batch. Reduction type is `"element"`.

## GenericPairLoss

```python
losses.GenericPairLoss(mat_based_loss, **kwargs)
```

**Parameters**:

- **mat_based_loss**: If `True`, the loss receives the distance matrix and pair masks; otherwise it receives the distances of the positive and negative pairs. See required implementations.

**Required Implementations**:

Implement the hook that matches `mat_based_loss`:

```python
# mat_based_loss=True: the full query-by-reference matrix and 0/1 pair masks
def _compute_matrix_loss(self, mat, pos_mask, neg_mask):
    raise NotImplementedError

# mat_based_loss=False: distances of the selected pairs and their indices
def _compute_pair_loss(self, pos_pairs, neg_pairs, indices_tuple):
    raise NotImplementedError
```

## GeneralizedLiftedStructureLoss

This was presented in [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737.pdf){target=_blank}. It is a modification of the original [LiftedStructureLoss](losses.md#liftedstructureloss)

```python
losses.GeneralizedLiftedStructureLoss(neg_margin=1, pos_margin=0, **kwargs)
```

**Equation**:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{LG}} = \sum_{i=1}^{P} \sum_{a=1}^{K} \Bigg[ & \log \sum_{\substack{p=1 \\ p \neq a}}^{K} e^{D\left(f(x_a^i),\, f(x_p^i)\right)} \\
& + \log \sum_{\substack{j=1 \\ j \neq i}}^{P} \sum_{n=1}^{K} e^{m - D\left(f(x_a^i),\, f(x_n^j)\right)} \Bigg]_+
\end{aligned}
$$

Here the batch has $P$ classes with $K$ samples each, $x_a^i$ is sample $a$ of class $i$, $f$ is the embedding network, and $D$ is the distance. The outer sums run over all anchors, the first inner sum over their positives, and the second over their negatives.

**Implementation notes**: The loss is computed per anchor and averaged over the batch rather than summed.

**Parameters**:

- **pos_margin**: The margin in the expression `e^(D - margin)`. The paper uses `pos_margin = 0 `, which is why this margin does not appear in the above equation.
- **neg_margin**: This is `m` in the above equation. The paper used values between 0.1 and 1.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. Reduction type is `"element"`.

## InstanceLoss

[Dual-Path Convolutional Image-Text Embeddings with Instance Loss](https://arxiv.org/pdf/1711.05535.pdf)

```python
losses.InstanceLoss(gamma=64, **kwargs)
```

**Parameters**:

- **gamma**: The cosine similarity matrix is scaled by this amount.

## HistogramLoss

[Learning Deep Embeddings with Histogram Loss](https://arxiv.org/pdf/1611.00822.pdf)

```python
losses.HistogramLoss(n_bins=None, delta=None)
```

**Parameters**:

- **n_bins**: The number of bins used to construct the histogram. Default is 100 when both `n_bins` and `delta` are `None`.
- **delta**: The mesh of the uniform partition of the interval [-1, 1] used to construct the histogram. If not set the value of n_bins will be used.

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)

**Default reducer**:

- This loss returns an **already reduced** loss.

## IntraPairVarianceLoss

[Deep Metric Learning with Tuplet Margin Loss](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Deep_Metric_Learning_With_Tuplet_Margin_Loss_ICCV_2019_paper.pdf){target=_blank}

```python
losses.IntraPairVarianceLoss(pos_eps=0.01, neg_eps=0.01, **kwargs)
```

**Equations**:

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

You should probably use this in conjunction with another loss, as described in the paper. You can accomplish this by using [MultipleLosses](losses.md#multiplelosses):

```python
main_loss = losses.TupletMarginLoss()
var_loss = losses.IntraPairVarianceLoss()
complete_loss = losses.MultipleLosses([main_loss, var_loss], weights=[1, 0.5])
```

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **pos_loss**: The loss per positive pair in the batch. Reduction type is `"pos_pair"`.
- **neg_loss**: The loss per negative pair in the batch. Reduction type is `"neg_pair"`.

## LargeMarginSoftmaxLoss

[Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/pdf/1612.02295.pdf){target=_blank}

```python
losses.LargeMarginSoftmaxLoss(num_classes, embedding_size, margin=4, scale=1,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **kwargs)
```

**Equations**:

$$
L_i = -\log\left(\frac{e^{\lVert \mathbf{W}_{y_i} \rVert \lVert \mathbf{x}_i \rVert \psi(\theta_{y_i})}}{e^{\lVert \mathbf{W}_{y_i} \rVert \lVert \mathbf{x}_i \rVert \psi(\theta_{y_i})} + \sum_{j \neq y_i} e^{\lVert \mathbf{W}_j \rVert \lVert \mathbf{x}_i \rVert \cos(\theta_j)}}\right)
$$

where

$$
\psi(\theta) = (-1)^k \cos(m\theta) - 2k, \quad \theta \in \left[\frac{k\pi}{m}, \frac{(k+1)\pi}{m}\right], \quad k \in \lbrace 0, 1, \ldots, m - 1 \rbrace
$$

$$
\cos(m\theta) = \sum_{n=0}^{\lfloor m/2 \rfloor} (-1)^n \binom{m}{2n} \cos^{m-2n}(\theta) \left(1 - \cos^2(\theta)\right)^n
$$

Here $\theta_j$ is the angle between $\mathbf{W}_j$ and $\mathbf{x}_i$. The implementation computes $\cos(m\theta)$ from $\cos\theta$ with this expansion, equations 6 and 7 of the paper, and treats $k$ as a constant for gradients.

**Implementation notes**: The paper's annealing of the target logit with a decreasing $\lambda$ is not implemented; schedule a combination with ordinary softmax in application code if needed.

**Parameters**:

- **num_classes**: The number of classes in your training dataset.
- **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set `embedding_size` to 512.
- **margin**: An integer which dictates the size of the angular margin. This is `m` in the above equation. The paper finds `m=4` works best.
- **scale**: The exponent multiplier in the loss's softmax expression. The paper uses `scale = 1 `, which is why it does not appear in the above equation.

**Other info**:

- Accepts the keyword-only [learnable weight arguments](#learnable-weight-arguments) `weight_regularizer`, `weight_reg_weight`, and `weight_init_func`.
- This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:

```python
loss_func = losses.LargeMarginSoftmaxLoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)

  - This is the only compatible distance.

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. Reduction type is `"element"`.

## LiftedStructureLoss

The original lifted structure loss as presented in [Deep Metric Learning via Lifted Structured Feature Embedding](https://arxiv.org/pdf/1511.06452.pdf){target=_blank}

```python
losses.LiftedStructureLoss(neg_margin=1, pos_margin=0, **kwargs):
```

**Equation**:

$$
\begin{aligned}
\tilde{J}_{i,j} &= \log\left(\sum_{(i,k) \in \mathcal{N}} \exp\lbrace \alpha - D_{i,k} \rbrace + \sum_{(j,l) \in \mathcal{N}} \exp\lbrace \alpha - D_{j,l} \rbrace\right) + D_{i,j} \\
\tilde{J} &= \frac{1}{2\lvert\mathcal{P}\rvert} \sum_{(i,j) \in \mathcal{P}} \max\left(0, \tilde{J}_{i,j}\right)^2
\end{aligned}
$$

Here $\mathcal{P}$ and $\mathcal{N}$ are the positive and negative pairs in the batch, and $D_{i,j}$ is the distance between embeddings $i$ and $j$.

**Implementation notes**: $\tilde{J}_{i,j}$ uses the negatives of both $i$ and $j$, so the anchors and positives must come from the same set. Passing a separate `ref_emb` raises `ValueError`; use [GeneralizedLiftedStructureLoss](#generalizedliftedstructureloss) for separate query and reference sets.

**Parameters**:

- **pos_margin**: The margin in the expression `D_(i,j) - margin`. The paper uses `pos_margin = 0 `, which is why it does not appear in the above equation.
- **neg_margin**: This is `alpha` in the above equation. The paper uses 1.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per positive pair in the batch. Reduction type is `"pos_pair"`.

## ManifoldLoss

[Ensemble Deep Manifold Similarity Learning using Hard Proxies](https://openaccess.thecvf.com/content_CVPR_2019/papers/Aziere_Ensemble_Deep_Manifold_Similarity_Learning_Using_Hard_Proxies_CVPR_2019_paper.pdf)

```python
losses.ManifoldLoss(
        embedding_size: int,
        num_proxies: int = 50,
        lambda_c: float = 1.0,
        alpha: float = 0.8,
        margin: float = 5e-4,
        **kwargs
    )
```

**Parameters**

- **embedding_size**: embedding size, $l$ in the paper.

- **num_proxies**: number of proxies, $K$ in the paper.

- **lambda_c**: regularization weight $\lambda_C$ in `loss = intrinsic_loss + lambda_c * context_loss`. `lambda_c=0` uses only the intrinsic loss, and `lambda_c=math.inf` uses only the context loss.

- **alpha**: parameter of the Random Walk. Must be in the range `(0,1)`. It specifies the amount of similarity between neighboring nodes.

- **margin**: margin used in the calculation of the loss.

Example usage:

```python
loss_fn = ManifoldLoss(128)

# use random cluster centers
loss = loss_fn(embeddings)
# or specify indices of embeddings to use as cluster centers
loss = loss_fn(embeddings, indices_tuple=indices)
```

**Important notes**

`labels`, `ref_emb`, and `ref_labels` are not supported for this loss function.

In addition, `indices_tuple` is **not** for the output of miners. Instead, it is for a list of indices of embeddings to be used as cluster centers.

**Default reducer**:

- This loss returns an **already reduced** loss.

## MarginLoss

[Sampling Matters in Deep Embedding Learning](https://arxiv.org/pdf/1706.07567.pdf){target=_blank}

```python
losses.MarginLoss(margin=0.2,
                nu=0,
                beta=1.2,
                triplets_per_anchor="all",
                learn_beta=False,
                num_classes=None,
                **kwargs)
```

**Equations**:

$$
\operatorname{minimize} \sum_{(i,j)} \ell^{\mathrm{margin}}(i,j) + \nu\left(\beta^{(0)} + \beta^{(\mathrm{class})}_{c(i)} + \beta^{(\mathrm{img})}_i\right)
$$

where

$$
\ell^{\mathrm{margin}}(i,j) = \left(\alpha + y_{ij}(D_{ij} - \beta)\right)_+
$$

Here $y_{ij}$ is $1$ for positive pairs and $-1$ for negative pairs, $D_{ij}$ is the distance between embeddings $i$ and $j$, and $c(i)$ is the class of sample $i$.

**Implementation notes**: The loss is computed over triplets, all of them by default, so each pair term repeats once per triplet that contains it. Positive and negative terms are divided by the number of nonzero terms. $\beta$ is either a single value or one value per class indexed by the anchor's class; the per-image $\beta^{(\mathrm{img})}$ is not supported.

**Parameters**:

- **margin**: This is alpha in the above equation. The paper uses 0.2.
- **nu**: The regularization weight for the magnitude of beta.
- **beta**: This is beta in the above equation. The paper uses 1.2 as the initial value.
- **triplets_per_anchor**: The number of triplets per element to sample within a batch. Can be an integer or the string "all". For example, if your batch size is 128, and triplets_per_anchor is 100, then 12800 triplets will be sampled. If triplets_per_anchor is "all", then all possible triplets in the batch will be used.
- **learn_beta**: If True, beta will be a torch.nn.Parameter, which can be optimized using any PyTorch optimizer.
- **num_classes**: If not None, then beta will be of size `num_classes`, so that a separate beta is used for each class during training.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

**Default reducer**:

- [DivisorReducer](reducers.md#divisorreducer)

**Reducer input**:

- **margin_loss**: The loss per triplet in the batch. Reduction type is `"triplet"`.
- **beta_reg_loss**: The regularization loss per element in `self.beta`. Reduction type is `"already_reduced"` if `self.num_classes = None`. Otherwise it is `"element"`.

## MultiSimilarityLoss

[Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf){target=_blank}

```python
losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5, **kwargs)
```

**Equation**:

$$
\begin{aligned}
\mathcal{L}_{MS} = \frac{1}{m} \sum_{i=1}^{m} \Bigg\lbrace & \frac{1}{\alpha} \log\left[1 + \sum_{k \in \mathcal{P}_i} e^{-\alpha(S_{ik} - \lambda)}\right] \\
& + \frac{1}{\beta} \log\left[1 + \sum_{k \in \mathcal{N}_i} e^{\beta(S_{ik} - \lambda)}\right] \Bigg\rbrace
\end{aligned}
$$

Here $m$ is the batch size, $S_{ik}$ is the similarity between embeddings $i$ and $k$, and $\mathcal{P}_i$ and $\mathcal{N}_i$ are the positive and negative pairs of anchor $i$.

**Implementation notes**: $\mathcal{P}_i$ and $\mathcal{N}_i$ are all positive and negative pairs of anchor $i$ unless a miner supplies them; the paper mines them with [MultiSimilarityMiner](miners.md#multisimilarityminer). The default `base=0.5` differs from the paper's $\lambda = 1$; pass `base=1` for the paper's setting.

**Parameters**:

- **alpha**: The weight applied to positive pairs. The paper uses 2.
- **beta**: The weight applied to negative pairs. The paper uses 50.
- **base**: The offset applied to the exponent in the loss. This is lambda in the above equation. The paper uses 1.

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. Reduction type is `"element"`.

## MultipleLosses

This is a simple wrapper for multiple losses. Pass in a list of already-initialized loss functions. Then, when you call forward on this object, it will return the sum of all wrapped losses.

```python
losses.MultipleLosses(losses, miners=None, weights=None)
```

**Parameters**:

- **losses**: A list or dictionary of initialized loss functions. On the forward call of MultipleLosses, each wrapped loss is computed and the weighted sum is returned.
- **miners**: Optional. A list or dictionary of mining functions. This allows you to pair mining functions with loss functions. For example, if `losses = [loss_A, loss_B]`, and `miners = [None, miner_B]` then no mining will be done for `loss_A`, but the output of `miner_B` will be passed to `loss_B`. The same logic applies if `losses = {"loss_A": loss_A, "loss_B": loss_B}` and `miners = {"loss_B": miner_B}`.
- **weights**: Optional. A list or dictionary of loss weights, which will be multiplied by the corresponding losses obtained by the loss functions. The default is to multiply each loss by 1. If `losses` is a list, then `weights` must be a list. If `losses` is a dictionary, `weights` must contain the same keys as `losses`.

## NCALoss

[Neighbourhood Components Analysis](https://www.cs.toronto.edu/~hinton/absps/nca.pdf){target=_blank}

```python
losses.NCALoss(softmax_scale=1, **kwargs)
```

**Equations**:

$$
g(A) = \sum_i \log\left(\sum_{j \in C_i} p_{ij}\right) = \sum_i \log(p_i)
$$

where

$$
p_i = \sum_{j \in C_i} p_{ij}
$$

$$
p_{ij} = \frac{\exp\left(-\lVert A x_i - A x_j \rVert^2\right)}{\sum_{k \neq i} \exp\left(-\lVert A x_i - A x_k \rVert^2\right)}
$$

Here $A x_i$ is the embedding of element $i$, $C_i$ is the set of elements with the same label as $i$, and $p_{ii} = 0$.

In this implementation, we use `-g(A)` as the loss.

**Implementation notes**: The loss is averaged over elements with $p_i > 0$; elements with no same-label reference are skipped. The default distance L2-normalizes embeddings, unlike the paper's linear map.

**Parameters**:

- **softmax_scale**: The exponent multiplier in the loss's softmax expression. The paper uses `softmax_scale = 1 `, which is why it does not appear in the above equations.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=2)`](distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch, that results in a non zero exponent in the cross entropy expression. Reduction type is `"element"`.

## NormalizedSoftmaxLoss

[Classification is a Strong Baseline for Deep Metric Learning](https://arxiv.org/pdf/1811.12649.pdf){target=_blank}

```python
losses.NormalizedSoftmaxLoss(num_classes, embedding_size, temperature=0.05,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **kwargs)
```

**Equation**:

$$
L_{\mathrm{norm}} = -\log\left(\frac{\exp\left(x^{T} p_{y} / \sigma\right)}{\sum_{z \in Z} \exp\left(x^{T} p_{z} / \sigma\right)}\right)
$$

Here $x$ is the L2-normalized embedding with label $y$, $p_z$ is the L2-normalized weight vector of class $z$, and $Z$ is the set of all classes.

**Parameters**:

- **num_classes**: The number of classes in your training dataset.
- **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set `embedding_size` to 512.
- **temperature**: This is sigma in the above equation. The paper uses 0.05.

**Other info**

- Accepts the keyword-only [learnable weight arguments](#learnable-weight-arguments) `weight_regularizer`, `weight_reg_weight`, and `weight_init_func`.
- This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:

```python
loss_func = losses.NormalizedSoftmaxLoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default distance**:

- [`DotProductSimilarity()`](distances.md#dotproductsimilarity)

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. Reduction type is `"element"`.

## NPairsLoss

[Improved Deep Metric Learning with Multi-class N-pair Loss Objective](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf){target=_blank}

If your batch has more than 2 samples per label, then you should use [NTXentLoss](#ntxentloss).

```python
losses.NPairsLoss(**kwargs)
```

**Default distance**:

- [`DotProductSimilarity()`](distances.md#dotproductsimilarity)

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. Reduction type is `"element"`.

## NTXentLoss

This is also known as InfoNCE, and is a generalization of the [NPairsLoss](losses.md#npairsloss). It has been used in self-supervision papers such as:

- [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf){target=_blank}
- [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/pdf/1911.05722.pdf){target=_blank}
- [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf){target=_blank}

**How exactly is the NTXentLoss computed?**

In the equation below, a loss is computed for each positive pair (`k_+`) in a batch, normalized by itself and all negative pairs in the batch that have the same "anchor" embedding (`k_i in K`).

- What does "anchor" mean? Let's say we have 3 pairs specified by batch indices: (0, 1), (0, 2), (1, 0). The first two pairs start with 0, so they have the same anchor. The third pair has the same indices as the first pair, but the order is different, so it does not have the same anchor.

Given `embeddings` with corresponding `labels`, positive pairs `(embeddings[i], embeddings[j])` are defined when `labels[i] == labels[j]`. Now let's look at an example loss calculation:

Consider `labels = [0, 0, 1, 2]`. Two losses will be computed:

- A positive pair of indices `[0, 1]`, with negative pairs of indices `[0, 2], [0, 3]`.

- A positive pair of indices `[1, 0]`, with negative pairs of indices `[1, 2], [1, 3]`.

Labels `1`, and `2` do not have positive pairs, and therefore the negative pair of indices `[2, 3]` will not be used.

Note that an anchor can belong to multiple positive pairs if its label is present multiple times in `labels`.

Are you trying to use `NTXentLoss` for self-supervised learning? Specifically, do you have two sets of embeddings which are derived from data that are augmented versions of each other? If so, you can skip the step of creating the `labels` array, by wrapping `NTXentLoss` with [`SelfSupervisedLoss`](losses.md#selfsupervisedloss).

```python
losses.NTXentLoss(temperature=0.07, **kwargs)
```

**Equation**:

$$
\mathcal{L}_q = -\log \frac{\exp\left(q \cdot k_+ / \tau\right)}{\sum_{i=0}^{K} \exp\left(q \cdot k_i / \tau\right)}
$$

Here $q$ is the anchor, $k_+$ its positive, $k_0, \ldots, k_K$ are $k_+$ plus the $K$ negatives sharing that anchor, and $q \cdot k$ is the similarity, cosine by default.

**Parameters**:

- **temperature**: This is tau in the above equation. The MoCo paper uses 0.07, while SimCLR uses 0.5.

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per positive pair in the batch. Reduction type is `"pos_pair"`.

## P2SGradLoss

[P2SGrad: Refined Gradients for Optimizing Deep Face Models](https://arxiv.org/abs/1905.02479)

```python
losses.P2SGradLoss(descriptors_dim, num_classes, **kwargs)
```

**Parameters**

- **descriptors_dim**: The embedding size.

- **num_classes**: The number of classes in your training dataset.

Example usage:

```python
loss_fn = P2SGradLoss(128, 10)
loss = loss_fn(embeddings, labels)
```

**Important notes**

`indices_tuple`, `ref_emb`, and `ref_labels` are not supported for this loss function.

**Default reducer**:

- This loss returns an **already reduced** loss.

## PNPLoss

[Rethinking the Optimization of Average Precision: Only Penalizing Negative Instances before Positive Ones is Enough](https://arxiv.org/pdf/2102.04640.pdf){target=_blank}

```python
losses.PNPLoss(b=2, alpha=1, anneal=0.01, variant="O", **kwargs)
```

**Equation**:

$$
\begin{aligned}
L_O &= \frac{1}{|S_P|} \sum_{i \in S_P} R(i, S_N) && (5) \\
L_{I_u} &= \frac{1}{|S_P|} \sum_{i \in S_P} \left(1 + R(i, S_N)\right) \log\left(1 + R(i, S_N)\right) && (7) \\
L_{I_b} &= \frac{1}{|S_P|} \sum_{i \in S_P} \frac{1}{b^2} \left(b R(i, S_N) - \log\left(1 + b R(i, S_N)\right)\right) && (9) \\
L_{D_s} &= \frac{1}{|S_P|} \sum_{i \in S_P} \log\left(1 + R(i, S_N)\right) && (12) \\
L_{D_q} &= 1 - \frac{1}{|S_P|} \sum_{i \in S_P} \frac{1}{\left(1 + R(i, S_N)\right)^{\alpha}} \quad (\alpha \geq 1) && (13)
\end{aligned}
$$

Here $R(i, S_N) = \sum_{j \in S_N} \sigma\left((s_j - s_i)/\tau\right)$, where $s_i$ is the cosine similarity between the query and element $i$, $\sigma$ is the sigmoid, and $\tau$ is `anneal`.

**Implementation notes**: The query is counted in its own $S_P$, as in the authors' implementation.

**Parameters**:

- **b**: The boundary of PNP-Ib (see equation 9 above). The paper uses 2.
- **alpha**: The power of PNP-Dq (see equation 13 above). The paper uses 8.
- **anneal**: The temperature of the sigmoid function. (The sigmoid function is used for `R` in the equations above.) The paper uses 0.01.
- **variant**: The name of the variant. The options are {"Ds", "Dq", "Iu", "Ib", "O"}. The paper uses "Dq".

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)
  - This is the only compatible distance.

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss averaged over elements that have at least 1 other positive in the batch. Reduction type is `"already_reduced"`, so the reducer does not change it.

## ProxyAnchorLoss

[Proxy Anchor Loss for Deep Metric Learning](https://arxiv.org/pdf/2003.13911.pdf){target=_blank}

```python
losses.ProxyAnchorLoss(num_classes, embedding_size, margin = 0.1, alpha = 32,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **kwargs)
```

**Equation**:

$$
\begin{aligned}
\ell(X) &= \frac{1}{|P^+|} \sum_{p \in P^+} \log\left(1 + \sum_{x \in X_p^+} e^{-\alpha\left(s(x, p) - \delta\right)}\right) \\
&\quad + \frac{1}{|P|} \sum_{p \in P} \log\left(1 + \sum_{x \in X_p^-} e^{\alpha\left(s(x, p) + \delta\right)}\right)
\end{aligned}
$$

Here $P$ is the set of all proxies, $P^+$ the proxies whose class is in the batch, $X_p^+$ and $X_p^-$ the batch embeddings with and without proxy $p$'s class, and $s(x, p)$ the cosine similarity.

**Parameters**:

- **num_classes**: The number of classes in your training dataset.
- **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set `embedding_size` to 512.
- **margin**: This is delta in the above equation. The paper uses 0.1.
- **alpha**: This is alpha in the above equation. The paper uses 32.

**Other info**

- Accepts the keyword-only [learnable weight arguments](#learnable-weight-arguments) `weight_regularizer`, `weight_reg_weight`, and `weight_init_func`.
- This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:

```python
loss_func = losses.ProxyAnchorLoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)

**Default reducer**:

- [DivisorReducer](reducers.md#divisorreducer)

**Reducer input**:

- **pos_loss**: The positive pair loss per proxy. Reduction type is `"element"`.
- **neg_loss**: The negative pair loss per proxy. Reduction type is `"element"`.

## ProxyNCALoss

[No Fuss Distance Metric Learning using Proxies](https://arxiv.org/pdf/1703.07464.pdf){target=_blank}

```python
losses.ProxyNCALoss(num_classes, embedding_size, softmax_scale=1,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **kwargs)
```

**Parameters**:

- **num_classes**: The number of classes in your training dataset.
- **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set `embedding_size` to 512.
- **softmax_scale**: See [NCALoss](losses.md#ncaloss)

**Other info**

- Accepts the keyword-only [learnable weight arguments](#learnable-weight-arguments) `weight_regularizer`, `weight_reg_weight`, and `weight_init_func`.
- This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:

```python
loss_func = losses.ProxyNCALoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=2)`](distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch, that results in a non zero exponent in the cross entropy expression. Reduction type is `"element"`.

## RankedListLoss

[Ranked List Loss for Deep Metric Learning](https://arxiv.org/abs/1903.03238)

```python
losses.RankedListLoss(margin, Tn, imbalance=0.5, alpha=None, Tp=0, **kwargs)
```

**Parameters**:

The defaults correspond to the paper's RLL-Simpler variant, which it recommends for exploratory analysis: `Tp=0` weights positives uniformly, and `alpha=None` sets $\alpha = 1 + m/2$ for margin $m$.

- **margin** (float): margin between positive and negative set
- **imbalance** (float): tradeoff between positive and negative sets. As the name suggests this takes into account
  the imbalance between positive and negative samples in the dataset
- **alpha** (float): smallest distance between negative points
- **Tp & Tn** (float): temperatures for, respectively, positive and negative pairs weighting.

## SelfSupervisedLoss

A common use case is to have `embeddings` and `ref_emb` be augmented versions of each other. For most losses, you have to create labels to indicate which `embeddings` correspond with which `ref_emb`.

`SelfSupervisedLoss` is a wrapper that takes care of this by creating labels internally. It assumes that:

- `ref_emb[i]` is an augmented version of `embeddings[i]`.
- `ref_emb[i]` is the only augmented version of `embeddings[i]` in the batch.

```python
losses.SelfSupervisedLoss(loss, symmetric=True, **kwargs)
```

**Parameters**:

- **loss**: The loss function to be wrapped.
- **symmetric**: If `True`, then the embeddings in both `embeddings` and `ref_emb` are used as anchors. If `False`, then only the embeddings in `embeddings` are used as anchors.

Example usage:

```
loss_fn = losses.TripletMarginLoss()
loss_fn = SelfSupervisedLoss(loss_fn)
loss = loss_fn(embeddings, ref_emb)
```

**Supported Loss Functions**

- [AngularLoss](losses.md#angularloss)
- [CircleLoss](losses.md#circleloss)
- [ContrastiveLoss](losses.md#contrastiveloss)
- [GeneralizedLiftedStructureLoss](losses.md#generalizedliftedstructureloss)
- [IntraPairVarianceLoss](losses.md#intrapairvarianceloss)
- [MultiSimilarityLoss](losses.md#multisimilarityloss)
- [NCALoss](losses.md#ncaloss)
- [NTXentLoss](losses.md#ntxentloss)
- [SignalToNoiseRatioContrastiveLoss](losses.md#signaltonoiseratiocontrastiveloss)
- [SupConLoss](losses.md#supconloss)
- [TripletMarginLoss](losses.md#tripletmarginloss)
- [TupletMarginLoss](losses.md#tupletmarginloss)

## SignalToNoiseRatioContrastiveLoss

[Signal-to-Noise Ratio: A Robust Distance Metric for Deep Metric Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf){target=_blank}

```python
losses.SignalToNoiseRatioContrastiveLoss(pos_margin=0, neg_margin=1, **kwargs):
```

**Parameters**:

- **pos_margin**: The noise-to-signal ratio over which positive pairs will contribute to the loss.
- **neg_margin**: The noise-to-signal ratio under which negative pairs will contribute to the loss.

**Default distance**:

- [`SNRDistance()`](distances.md#snrdistance)
  - This is the only compatible distance.

**Default reducer**:

- [AvgNonZeroReducer](reducers.md#avgnonzeroreducer)

**Reducer input**:

- **pos_loss**: The loss per positive pair in the batch. Reduction type is `"pos_pair"`.
- **neg_loss**: The loss per negative pair in the batch. Reduction type is `"neg_pair"`.

## SmoothAPLoss

[Smooth-AP: Smoothing the Path Towards Large-Scale Image Retrieval](https://arxiv.org/abs/2007.12163){target=_blank}

```python
losses.SmoothAPLoss(
    temperature=0.01,
    **kwargs
)
```

**Equations**:

$$
\mathcal{G}(x; \tau) = \frac{1}{1 + e^{-x / \tau}}
$$

$$
AP_q \approx \frac{1}{|\mathcal{S}_P|} \sum_{i \in \mathcal{S}_P} \frac{1 + \sum_{j \in \mathcal{S}_P, j \neq i} \mathcal{G}(D_{ij}; \tau)}{1 + \sum_{j \in \mathcal{S}_P, j \neq i} \mathcal{G}(D_{ij}; \tau) + \sum_{j \in \mathcal{S}_N} \mathcal{G}(D_{ij}; \tau)}
$$

$$
\mathcal{L}_{AP} = \frac{1}{m} \sum_{k=1}^{m} \left(1 - AP_k\right)
$$

Here $D_{ij} = s_j - s_i$, where $s_i$ is the cosine similarity between query $q$ and element $i$; $\mathcal{S}_P$ and $\mathcal{S}_N$ are the positive and negative sets of $q$; and $m$ is the batch size.

**Implementation notes**: The query is counted in its own $\mathcal{S}_P$, as in the authors' implementation.

**Parameters**:

- **temperature**: The desired temperature for scaling the sigmoid function. This is denoted by $\tau$ in the first and second equations.

**Other info**:

- Positive sets come from the labels, so classes may appear in any order and with different counts. Each element is a query against the rest of the batch. Memory grows with the cube of the batch size.

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)
  - This is the only compatible distance.

## SoftTripleLoss

[SoftTriple Loss: Deep Metric Learning Without Triplet Sampling](http://openaccess.thecvf.com/content_ICCV_2019/papers/Qian_SoftTriple_Loss_Deep_Metric_Learning_Without_Triplet_Sampling_ICCV_2019_paper.pdf){target=_blank}

```python
losses.SoftTripleLoss(num_classes, embedding_size, centers_per_class=10, la=20, gamma=0.1, margin=0.01,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **kwargs)
```

**Equations**:

$$
\ell_{\operatorname{SoftTriple}}(\mathbf{x}_i) = -\log \frac{\exp\left(\lambda\left(\mathcal{S}'_{i,y_i}-\delta\right)\right)}{\exp\left(\lambda\left(\mathcal{S}'_{i,y_i}-\delta\right)\right)+\sum_{j \neq y_i}\exp\left(\lambda \mathcal{S}'_{i,j}\right)}
$$

where

$$
\mathcal{S}'_{i,c} = \sum_{k} \frac{\exp\left(\frac{1}{\gamma}\mathbf{x}_i^\top \mathbf{w}_c^k\right)}{\sum_{k'}\exp\left(\frac{1}{\gamma}\mathbf{x}_i^\top \mathbf{w}_c^{k'}\right)} \mathbf{x}_i^\top \mathbf{w}_c^k
$$

Here $\mathbf{w}_c^k$ is the $k$-th of the `centers_per_class` centers of class $c$; `CosineSimilarity` L2-normalizes both $\mathbf{x}_i$ and $\mathbf{w}_c^k$.

**Implementation notes**: The paper adds a center regularizer with weight $\tau = 0.2$. Reproduce it with `weight_regularizer=regularizers.SparseCentersRegularizer(num_classes, centers_per_class)` and `weight_reg_weight=0.2`; it is not enabled by default.

**Parameters**:

- **num_classes**: The number of classes in your training dataset.
- **embedding_size**: The size of the embeddings that you pass into the loss function. For example, if your batch size is 128 and your network outputs 512 dimensional embeddings, then set `embedding_size` to 512.
- **centers_per_class**: The number of weight vectors per class. (The regular cross entropy loss has 1 center per class.) The paper uses 10.
- **la**: This is lambda in the above equation.
- **gamma**: This is gamma in the above equation. The paper uses 0.1.
- **margin**: This is delta in the above equations. The paper uses 0.01.

**Other info**

- Accepts the keyword-only [learnable weight arguments](#learnable-weight-arguments) `weight_regularizer`, `weight_reg_weight`, and `weight_init_func`.
- This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:

```python
loss_func = losses.SoftTripleLoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)
  - The distance measure must be inverted. For example, [`DotProductSimilarity(normalize_embeddings=False)`](distances.md#dotproductsimilarity) is also compatible.

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. Reduction type is `"element"`.

## SphereFaceLoss

[SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/pdf/1704.08063.pdf){target=_blank}

```python
losses.SphereFaceLoss(num_classes, embedding_size, margin=4, scale=1,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **kwargs)
```

**Parameters**:

See [LargeMarginSoftmaxLoss](losses.md#largemarginsoftmaxloss)

**Other info**

- Accepts the keyword-only [learnable weight arguments](#learnable-weight-arguments) `weight_regularizer`, `weight_reg_weight`, and `weight_init_func`.
- This loss **requires an optimizer**. You need to create an optimizer and pass this loss's parameters to that optimizer. For example:

```python
loss_func = losses.SphereFaceLoss(...).to(torch.device('cuda'))
loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
# then during training:
loss_optimizer.step()
```

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)

  - This is the only compatible distance.

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. Reduction type is `"element"`.

## SubCenterArcFaceLoss

[Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf){target=_blank}

This loss extends [ArcFaceLoss](losses.md#arcfaceloss). It uses multiple sub centers per class, instead of just a single center, hence the name Sub-center ArcFace.

```python
losses.SubCenterArcFaceLoss(num_classes, embedding_size, margin=28.6, scale=64, sub_centers=3,
    *, weight_regularizer=None, weight_reg_weight=1, weight_init_func=None, **kwargs)
```

**Parameters**:

- **sub_centers**: The number of sub centers per class.

See [ArcFaceLoss](losses.md#arcfaceloss) for a description of the other parameters.

**Other info**:

- This loss **requires an optimizer**. See [ArcFaceLoss](losses.md#arcfaceloss) for details.
- See [ArcFaceLoss](losses.md#arcfaceloss) for default distance, reducer, and reducer input.

**Getting outliers and dominant centers**

Outliers and dominant centers can be computed as described in the paper.

```python
outliers, dominant_centers = loss_func.get_outliers(
    embeddings, labels, threshold=75, return_dominant_centers=True
)
```

## SupConLoss

Described in [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362){target=_blank}.

```python
losses.SupConLoss(temperature=0.1, **kwargs)
```

**Equation**:

$$
\mathcal{L}^{sup}_{out} = \sum_{i \in I} \frac{-1}{\left|P(i)\right|} \sum_{p \in P(i)} \log \frac{\exp\left(\mathbf{z}_i \cdot \mathbf{z}_p / \tau\right)}{\sum_{a \in A(i)} \exp\left(\mathbf{z}_i \cdot \mathbf{z}_a / \tau\right)}
$$

Here $I$ indexes the batch, $A(i)$ is every index except $i$, and $P(i) \subseteq A(i)$ holds the indices that share $i$'s label.

**Implementation notes**: The default reducer averages over anchors with a positive loss instead of summing. A batch without both a positive and a negative pair yields 0.

**Parameters**:

- **temperature**: This is tau in the above equation. The paper uses 0.1.

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)

**Default reducer**:

- [AvgNonZeroReducer](reducers.md#avgnonzeroreducer)

**Reducer input**:

- **loss**: The loss per element in the batch. If an element has only negative pairs or no pairs, it's ignored thanks to `AvgNonZeroReducer`. Reduction type is `"element"`.

## ThresholdConsistentMarginLoss

[Threshold-Consistent Margin Loss for Open-World Deep Metric Learning](https://arxiv.org/pdf/2307.04047){target=_blank}

This loss acts as a form of regularization and is usually combined with another metric loss function.

```python
losses.ThresholdConsistentMarginLoss(
    lambda_plus=1.0,
    lambda_minus=1.0,
    margin_plus=0.9,
    margin_minus=0.5,
    **kwargs
)
```

**Equation**:

$$
L_{\operatorname{TCM}} = \lambda^+ \cdot \frac{\sum_{s \in S^+} \left(m^+ - s\right) \cdot \mathbf{1}_{s \le m^+}}{\sum_{s \in S^+} \mathbf{1}_{s \le m^+}} + \lambda^- \cdot \frac{\sum_{s \in S^-} \left(s - m^-\right) \cdot \mathbf{1}_{s \ge m^-}}{\sum_{s \in S^-} \mathbf{1}_{s \ge m^-}}
$$

Here $S^+$ and $S^-$ are the cosine similarities of the batch's positive and negative pairs. A term whose denominator is zero contributes 0.

**Parameters**:

- **lambda_plus**: The scaling coefficient for the anchor-positive part of the loss. This is $\lambda^+$ in the above equation.
- **lambda_minus**: The scaling coefficient for the anchor-negative part of the loss. This is $\lambda^-$ in the above equation.
- **margin_plus**: The minimum anchor-positive similarity to be included in the loss. This is $m^+$ in the above equation.
- **margin_minus**: The maximum anchor-negative similarity to be included in the loss. This is $m^-$ in the above equation.

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)
  - This is the only compatible distance.

## TripletMarginLoss

```python
losses.TripletMarginLoss(margin=0.05,
                        swap=False,
                        smooth_loss=False,
                        triplets_per_anchor="all",
                        **kwargs)
```

**Equation**:

$$
L_{triplet} = \left[d_{ap} - d_{an} + m\right]_+
$$

Here $d_{ap}$ and $d_{an}$ are the anchor-positive and anchor-negative distances, and $[x]_+ = \max(0, x)$. With a similarity such as `CosineSimilarity`, the loss is $\left[s_{an} - s_{ap} + m\right]_+$.

**Parameters**:

- **margin**: The desired difference between the anchor-positive distance and the anchor-negative distance. This is `m` in the above equation.
- **swap**: Use the positive-negative distance instead of anchor-negative distance, if it violates the margin more.
- **smooth_loss**: Use the log-exp version of the triplet loss
- **triplets_per_anchor**: The number of triplets per element to sample within a batch. Can be an integer or the string "all". For example, if your batch size is 128, and triplets_per_anchor is 100, then 12800 triplets will be sampled. If triplets_per_anchor is "all", then all possible triplets in the batch will be used.

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](distances.md#lpdistance)

**Default reducer**:

- [AvgNonZeroReducer](reducers.md#avgnonzeroreducer)

**Reducer input**:

- **loss**: The loss per triplet in the batch. Reduction type is `"triplet"`.

## TupletMarginLoss

[Deep Metric Learning with Tuplet Margin Loss](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Deep_Metric_Learning_With_Tuplet_Margin_Loss_ICCV_2019_paper.pdf){target=_blank}

```python
losses.TupletMarginLoss(margin=5.73, scale=64, **kwargs)
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

The paper combines this loss with [IntraPairVarianceLoss](losses.md#intrapairvarianceloss). You can accomplish this by using [MultipleLosses](losses.md#multiplelosses):

```python
main_loss = losses.TupletMarginLoss()
var_loss = losses.IntraPairVarianceLoss()
complete_loss = losses.MultipleLosses([main_loss, var_loss], weights=[1, 0.5])
```

**Default distance**:

- [`CosineSimilarity()`](distances.md#cosinesimilarity)

  - This is the only compatible distance.

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per positive pair in the batch. Reduction type is `"pos_pair"`.

## Learnable weight arguments

Losses that own class weights or proxies declare these keyword-only arguments
in their signatures:

- **weight_regularizer**: A [regularizer](regularizers.md) applied to the loss's
  learnable weights, one row per class vector or proxy. If `None`, no weight
  penalty is added.
- **weight_reg_weight**: The weight multiplying that penalty before it is added
  to the reduced loss.
- **weight_init_func**: A callable that initializes the parameter in place, such
  as `torch.nn.init.xavier_uniform_` or
  `functools.partial(torch.nn.init.kaiming_uniform_, a=1, mode="fan_out")`. If
  `None`, the loss's default below is used.

| Loss                                                                                                                                                                                          | Parameter           | Default initialization                           |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------ |
| [ArcFaceLoss](#arcfaceloss), [CosFaceLoss](#cosfaceloss), [LargeMarginSoftmaxLoss](#largemarginsoftmaxloss), [SphereFaceLoss](#spherefaceloss), [SubCenterArcFaceLoss](#subcenterarcfaceloss) | `W`, `(D, C)`       | `torch.nn.init.normal_`                          |
| [NormalizedSoftmaxLoss](#normalizedsoftmaxloss)                                                                                                                                               | `W`, `(D, C)`       | `torch.nn.init.normal_`                          |
| [ProxyAnchorLoss](#proxyanchorloss)                                                                                                                                                           | `proxies`, `(C, D)` | `torch.nn.init.kaiming_normal_(mode="fan_out")`  |
| [ProxyNCALoss](#proxyncaloss)                                                                                                                                                                 | `proxies`, `(C, D)` | `torch.nn.init.normal_`                          |
| [SoftTripleLoss](#softtripleloss)                                                                                                                                                             | `fc`, `(D, C·K)`    | `torch.nn.init.kaiming_uniform_(a=math.sqrt(5))` |

`D` is the embedding size, `C` the number of classes, and `K` the centers per
class. `SubCenterArcFaceLoss` stores `C·sub_centers` columns.

These parameters belong to the loss module. Move the loss with
`loss_func.to(device, dtype)` and include `loss_func.parameters()` in the
optimizer. Outside autocast, the parameters must have the embeddings' device and
dtype; otherwise the loss raises `ValueError` naming both placements. Under
autocast, parameters may keep a higher precision than the embeddings. The loss
never moves or casts its own parameters during a forward pass.
`P2SGradLoss` and `ManifoldLoss` own parameters with the same placement rule;
their regularization arguments are only the embedding ones.

## VICRegLoss

[VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/pdf/2105.04906.pdf){target=_blank}

```python
losses.VICRegLoss(invariance_lambda=25,
                variance_mu=25,
                covariance_v=1,
                eps=1e-4,
                **kwargs)
```

**Usage**:

Unlike other loss functions, `VICRegLoss` does not accept `labels` or `indices_tuple`:

```python
loss_fn = VICRegLoss()
loss = loss_fn(embeddings, ref_emb=ref_emb)
```

**Equations**:

$$
\ell(Z, Z') = \lambda s(Z, Z') + \mu\left[v(Z) + v(Z')\right] + \nu\left[c(Z) + c(Z')\right]
$$

Here $\lambda$, $\mu$, and $\nu$ are `invariance_lambda`, `variance_mu`, and `covariance_v`; $n$ is the batch size and $d$ the embedding size.

where

$$
s(Z, Z') = \frac{1}{n} \sum_{i} \left\lVert z_i - z'_i \right\rVert_2^2
$$

$$
v(Z) = \frac{1}{d} \sum_{j=1}^{d} \max\left(0, \gamma - S\left(z^j, \epsilon\right)\right)
$$

Here $z^j$ is the $j$-th embedding dimension across the batch, $\gamma = 1$, and $\epsilon$ is `eps`.

$$
S(x, \epsilon) = \sqrt{\operatorname{Var}(x) + \epsilon}
$$

$$
c(Z) = \frac{1}{d} \sum_{i \neq j} \left[C(Z)\right]_{i,j}^2
$$

Here $C(Z)$ is the covariance matrix of $Z$, normalized by $n - 1$.

**Implementation notes**: The invariance term is a mean squared error over both rows and dimensions, $\frac{1}{nd}\sum_i \lVert z_i - z'_i \rVert_2^2$, as in the paper's pseudocode and the authors' implementation. The variance term is $\frac{\mu}{2}\left[v(Z) + v(Z')\right]$, also following the authors' implementation; $\gamma$ is fixed at 1.

**Parameters**:

- **invariance_lambda**: The weight of the invariance term.
- **variance_mu**: The weight of the variance term.
- **covariance_v**: The weight of the covariance term.
- **eps**: Small scalar to prevent numerical instability.

**Default distance**:

- Not applicable. You cannot pass in a distance function.

**Default reducer**:

- [MeanReducer](reducers.md#meanreducer)

**Reducer input**:

- **invariance_loss**: The MSE loss between `embeddings[i]` and `ref_emb[i]`. Reduction type is `"element"`.
- **variance_loss1**: The variance loss for `embeddings`. Reduction type is `"element"`.
- **variance_loss2**: The variance loss for `ref_emb`. Reduction type is `"element"`.
- **covariance_loss**: The covariance loss. This loss is already reduced to a single value.
