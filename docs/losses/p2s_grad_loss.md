# P2SGradLoss

[P2SGrad: Refined Gradients for Optimizing Deep Face Models](https://arxiv.org/abs/1905.02479)

**Mathematical reference**: The paper linked above defines the objective and derivation. This page specifies the library interface and its settings.

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
