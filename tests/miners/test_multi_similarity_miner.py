import unittest
import torch
from pytorch_metric_learning.miners import MultiSimilarityMiner
from pytorch_metric_learning.utils import common_functions as c_f

class TestMultiSimilarityMiner(unittest.TestCase):
    def test_multi_similarity_miner(self):
        epsilon = 0.1
        miner = MultiSimilarityMiner(epsilon)
        embedding_angles = torch.arange(0, 64)
        embeddings = torch.tensor([c_f.angle_to_coord(a) for a in embedding_angles], requires_grad=True, dtype=torch.float) #2D embeddings
        labels = torch.randint(low=0, high=10, size=(64,))
        pos_pairs = []
        neg_pairs = []
        for i in range(len(embeddings)):
            anchor, anchor_label = embeddings[i], labels[i]
            for j in range(len(embeddings)):
                if j != i:
                    other, other_label = embeddings[j], labels[j]
                    if anchor_label == other_label:
                        pos_pairs.append((i,j,torch.matmul(anchor, other.t()).item()))
                    if anchor_label != other_label:
                        neg_pairs.append((i,j,torch.matmul(anchor, other.t()).item()))
        
        correct_a1, correct_p = [], []
        correct_a2, correct_n = [], []
        for a1,p,ap_sim in pos_pairs:
            max_neg_sim = float('-inf')
            for a2,n,an_sim in neg_pairs:
                if a2==a1:
                    if an_sim > max_neg_sim:
                        max_neg_sim = an_sim
            if ap_sim < max_neg_sim + epsilon:
                correct_a1.append(a1)
                correct_p.append(p)

        for a2,n,an_sim in neg_pairs:
            min_pos_sim = float('inf')
            for a1,p,ap_sim in pos_pairs:
                if a2==a1:
                    if ap_sim < min_pos_sim:
                        min_pos_sim = ap_sim
            if an_sim > min_pos_sim - epsilon:
                correct_a2.append(a2)
                correct_n.append(n)

        correct_pos_pairs = set([(a,p) for a,p in zip(correct_a1, correct_p)])
        correct_neg_pairs = set([(a,n) for a,n in zip(correct_a2, correct_n)])

        a1, p1, a2, n2 = miner(embeddings, labels)
        pos_pairs = set([(a.item(),p.item()) for a,p in zip(a1,p1)])
        neg_pairs = set([(a.item(),n.item()) for a,n in zip(a2,n2)])

        self.assertTrue(pos_pairs == correct_pos_pairs)
        self.assertTrue(neg_pairs == correct_neg_pairs)


    def test_empty_output(self):
        miner = MultiSimilarityMiner(0.1)
        batch_size = 32
        embeddings = torch.randn(batch_size, 64)
        labels = torch.arange(batch_size)
        a1, p, _, _ = miner(embeddings, labels)
        self.assertTrue(len(a1)==0)
        self.assertTrue(len(p)==0)