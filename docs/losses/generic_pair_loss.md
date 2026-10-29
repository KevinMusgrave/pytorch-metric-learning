# GenericPairLoss

```python
losses.GenericPairLoss(**options)
```

A base class for losses over positive and negative pairs. Its `compute_loss`
converts labels or a supplied `indices_tuple` to pairs, returns zero when there
are none, computes the distance matrix between `embeddings` and `ref_emb`, and
calls `pair_loss`.

**Required Implementations**:

```python
def pair_loss(self, mat, pairs):
    raise NotImplementedError
```

`mat` is the `(len(embeddings), len(ref_emb))` distance matrix and `pairs` is
the four-tensor `(a1, p, a2, n)` tuple. Two static helpers extract the common
views: `self.pair_values(mat, pairs)` returns the positive-pair and
negative-pair distances, and `self.pair_masks(mat, pairs)` returns boolean
positive and negative masks over `mat`. See
[pair-based losses](../extend/losses.md#pair-based-losses) for an example.
