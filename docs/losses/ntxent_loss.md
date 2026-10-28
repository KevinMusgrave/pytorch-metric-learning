# NTXentLoss

This is also known as InfoNCE, and is a generalization of the [NPairsLoss](n_pairs_loss.md#npairsloss). It has been used in self-supervision papers such as:

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

Are you trying to use `NTXentLoss` for self-supervised learning? Specifically, do you have two sets of embeddings which are derived from data that are augmented versions of each other? If so, you can skip the step of creating the `labels` array, by wrapping `NTXentLoss` with [`SelfSupervisedLoss`](self_supervised_loss.md#selfsupervisedloss).

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

- [`CosineSimilarity()`](../distances.md#cosinesimilarity)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)

**Reducer input**:

- **loss**: The loss per positive pair in the batch. Reduction type is `"pos_pair"`.
