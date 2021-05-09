import torch


def get_triplet_embeddings_with_ref(dtype, device):
    embeddings = torch.randn(4, 32, requires_grad=True, device=device, dtype=dtype)
    embeddings = torch.nn.functional.normalize(embeddings)
    labels = torch.LongTensor([0, 0, 1, 1])

    ref_emb = torch.randn(3, 32, requires_grad=True, device=device, dtype=dtype)
    ref_emb = torch.nn.functional.normalize(ref_emb)
    ref_labels = torch.LongTensor([0, 1, 2])

    triplets = [
        (0, 0, 1),
        (0, 0, 2),
        (1, 0, 1),
        (1, 0, 2),
        (2, 1, 0),
        (2, 1, 2),
        (3, 1, 0),
        (3, 1, 2),
    ]

    return embeddings, labels, ref_emb, ref_labels, triplets
