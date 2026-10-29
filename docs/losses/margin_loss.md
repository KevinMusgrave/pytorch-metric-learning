# MarginLoss

[Sampling Matters in Deep Embedding Learning](https://arxiv.org/pdf/1706.07567.pdf){target=_blank}

```python
losses.MarginLoss(margin=0.2,
                nu=0,
                beta=1.2,
                triplets_per_anchor="all",
                learn_beta=False,
                num_classes=None,
                **options)
```

**Equation**:

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

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance)

**Default reducer**:

- [DivisorReducer](../reducers.md#divisorreducer)

**Reducer input**:

- **margin_loss**: The loss per triplet in the batch. Reduction type is `"triplet"`.
- **beta_reg_loss**: The regularization loss per element in `self.beta`. Reduction type is `"already_reduced"` if `self.num_classes = None`. Otherwise it is `"element"`.
