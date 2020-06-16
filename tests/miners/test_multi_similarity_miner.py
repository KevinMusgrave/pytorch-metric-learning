import unittest
import torch
from pytorch_metric_learning.miners import MultiSimilarityMiner
from pytorch_metric_learning.utils import common_functions as c_f

class TestMultiSimilarityMiner(unittest.TestCase):
    def test_multi_similarity_miner(self):
        epsilon = 0.1
        miner = MultiSimilarityMiner(epsilon)
        angles = [0, 20, 40, 60, 80, 100, 120, 140, 160]
        embeddings = torch.FloatTensor([c_f.angle_to_coord(a) for a in angles])
        labels = torch.LongTensor([0, 0, 1, 1, 0, 2, 1, 1, 1])
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

        correct_a1 = torch.LongTensor(correct_a1)
        correct_p = torch.LongTensor(correct_p)
        correct_a2 = torch.LongTensor(correct_a2)
        correct_n = torch.LongTensor(correct_n)

        a1, p, a2, n = miner(embeddings, labels)

        for unique_a1 in torch.unique(correct_a1):
            positives = p[torch.where(a1==unique_a1)[0]].cpu().numpy()
            correct_positives = correct_p[torch.where(correct_a1==unique_a1)[0]].cpu().numpy()
            self.assertTrue(set(positives) == set(correct_positives))

        for unique_a2 in torch.unique(correct_a2):
            negatives = n[torch.where(a2==unique_a2)[0]].cpu().numpy()
            correct_negatives = correct_n[torch.where(correct_a2==unique_a2)[0]].cpu().numpy()
            self.assertTrue(set(negatives) == set(correct_negatives))


    def test_empty_output(self):
        miner = MultiSimilarityMiner(0.1)
        batch_size = 32
        embeddings = torch.randn(batch_size, 64)
        labels = torch.arange(batch_size)
        a1, p, _, _ = miner(embeddings, labels)
        self.assertTrue(len(a1)==0)
        self.assertTrue(len(p)==0)

if __name__ == '__main__':
    unittest.main()