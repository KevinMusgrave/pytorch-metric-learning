# GenericPairLoss

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
