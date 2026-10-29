# GenericPairLoss

A base class for losses over positive and negative pairs.

```python
losses.GenericPairLoss(**options)
```

**Implementation notes**: Its `compute_loss` converts labels or a supplied
`indices_tuple` to `types.PairIndices`, returns `{}` when there are no pairs,
computes the distance matrix between `embeddings` and the reference, and calls
`pair_loss`. Labels or an `indices_tuple` are required; with neither, it raises
`ValueError`. Subclasses implement:

```python
def pair_loss(self, mat, pairs):
    raise NotImplementedError
```

`mat` is the `(N, N)` distance matrix of `embeddings` with itself when `ref_emb`
is `None`, or the `(N, M)` matrix between `embeddings` and `ref_emb`. Without
`ref_emb`, an embedding is never its own positive. `pairs` is a `PairIndices`
`(a1, p, a2, n)`. Two static helpers extract the common views:
`self.pair_values(mat, pairs)` returns the positive-pair and negative-pair
distances, and `self.pair_masks(mat, pairs)` returns boolean positive and
negative masks over `mat`. `pair_loss` returns a `types.LossDict`; see
[pair-based losses](../extend/losses.md#pair-based-losses) for an example.

**Parameters**:

- **options**: The [loss options](../losses.md#loss-options).

**Default distance**:

- [`LpDistance(normalize_embeddings=True, p=2, power=1)`](../distances.md#lpdistance)

**Default reducer**:

- [MeanReducer](../reducers.md#meanreducer)
