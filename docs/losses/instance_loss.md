# InstanceLoss

[Dual-Path Convolutional Image-Text Embeddings with Instance Loss](https://arxiv.org/pdf/1711.05535.pdf)

**Mathematical reference**: The paper linked above defines the objective and derivation. This page specifies the library interface and its settings.

```python
losses.InstanceLoss(gamma=64, **kwargs)
```

**Parameters**:

- **gamma**: The cosine similarity matrix is scaled by this amount.
