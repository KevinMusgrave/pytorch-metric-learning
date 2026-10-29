# InstanceLoss

[Dual-Path Convolutional Image-Text Embeddings with Instance Loss](https://arxiv.org/pdf/1711.05535.pdf)

```python
losses.InstanceLoss(gamma=64, **options)
```

**Equation**:

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp\left(\gamma \cos(\mathbf{x}_i, \mathbf{x}_{y_i})\right)}{\sum_{j=1}^{N} \exp\left(\gamma \cos(\mathbf{x}_i, \mathbf{x}_j)\right)}
$$

Each batch element is treated as its own class and classified against every element in the batch, with $\mathbf{x}_{y_i}$ the element whose label matches $i$'s.

**Implementation notes**: Labels are remapped to `0..K-1` within the batch, and the target column for element $i$ is the first batch position of its label. The paper's formulation assumes one instance per label in the batch.

**Parameters**:

- **gamma**: $\gamma$, the scale applied to the cosine similarity matrix. The paper uses 64 for its two-stream setting.
