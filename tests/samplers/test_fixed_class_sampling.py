from collections import Counter

import pytest
import torch

from pytorch_metric_learning.samplers import FixedSetOfPairs, FixedSetOfTriplets


@pytest.mark.parametrize("count", [1, 2, 7, 1000])
def test_pair_classes_members_and_seeded_replay(count):
    labels = torch.tensor([0, 1, 1, 2, 2, 2])
    first = FixedSetOfPairs(
        labels, count, batch_size=6, generator=torch.Generator().manual_seed(71)
    )
    second = FixedSetOfPairs(
        labels, count, batch_size=6, generator=torch.Generator().manual_seed(71)
    )
    pairs = first.fixed_set_of_pairs
    assert torch.equal(pairs, second.fixed_set_of_pairs)
    positive = (count + 1) // 2
    assert torch.equal(labels[pairs[:positive, 0]], labels[pairs[:positive, 1]])
    assert torch.all(pairs[:positive, 0] != pairs[:positive, 1])
    assert torch.all(labels[pairs[positive:, 0]] != labels[pairs[positive:, 1]])
    assert Counter(
        tuple(pair)
        for batch in first
        for pair in torch.tensor(batch).view(-1, 2).tolist()
    ) == Counter(map(tuple, pairs.tolist()))


@pytest.mark.parametrize("count", [1, 7, 1000])
def test_triplet_classes_members_and_seeded_replay(count):
    labels = torch.tensor([0, 1, 1, 2, 2, 2])
    first = FixedSetOfTriplets(
        labels, count, generator=torch.Generator().manual_seed(91)
    )
    second = FixedSetOfTriplets(
        labels, count, generator=torch.Generator().manual_seed(91)
    )
    triples = first.fixed_set_of_triplets
    assert torch.equal(triples, second.fixed_set_of_triplets)
    a, p, n = triples.unbind(dim=1)
    assert torch.equal(labels[a], labels[p])
    assert torch.all(labels[a] != labels[n])
    assert torch.all(a[labels[a] != 0] != p[labels[a] != 0])
    assert torch.all(a[labels[a] == 0] == p[labels[a] == 0])
    assert Counter(
        map(tuple, torch.tensor(list(first)).view(-1, 3).tolist())
    ) == Counter(map(tuple, triples.tolist()))


@pytest.mark.parametrize("kind", ["pairs", "triplets"])
def test_ordered_negative_class_pairs_are_uniform_with_unequal_classes(kind):
    labels = torch.repeat_interleave(torch.arange(5), torch.tensor([1, 2, 3, 4, 5]))
    generator = torch.Generator().manual_seed(123)
    if kind == "pairs":
        sampler = FixedSetOfPairs(labels, 40000, batch_size=64, generator=generator)
        selected = sampler.fixed_set_of_pairs[20000:]
    else:
        sampler = FixedSetOfTriplets(labels, 20000, generator=generator)
        selected = sampler.fixed_set_of_triplets[:, [0, 2]]
    counts = Counter(map(tuple, labels[selected].tolist()))
    assert set(counts) == {(a, b) for a in range(5) for b in range(5) if a != b}
    assert all(abs(count - 1000) < 250 for count in counts.values())


@pytest.mark.parametrize("kind", ["pairs", "triplets"])
def test_explicit_generators_do_not_advance_the_global_rng(kind):
    state = torch.random.get_rng_state().clone()
    labels = torch.arange(6).repeat_interleave(2)
    generator = torch.Generator().manual_seed(12)
    if kind == "pairs":
        list(FixedSetOfPairs(labels, 21, batch_size=8, generator=generator))
    else:
        list(FixedSetOfTriplets(labels, 21, generator=generator))
    assert torch.equal(torch.random.get_rng_state(), state)
