import unittest
import torch
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.losses import CrossBatchMemory, ContrastiveLoss
from pytorch_metric_learning.miners import PairMarginMiner, TripletMarginMiner

class TestCrossBatchMemory(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.embedding_size = 128
        self.memory_size = 321

    def test_queue(self):
        batch_size = 32
        self.loss = CrossBatchMemory(loss=ContrastiveLoss(), embedding_size=self.embedding_size, memory_size=self.memory_size)
        for i in range(30):
            embeddings = torch.randn(batch_size, self.embedding_size)
            labels = torch.arange(batch_size)
            q = self.loss.queue_idx
            self.assertTrue(q==(i*batch_size)%self.memory_size)
            loss = self.loss(embeddings, labels)

            start_idx = q
            if q+batch_size == self.memory_size:
                end_idx = self.memory_size
            else:
                end_idx = (q+batch_size)%self.memory_size
            if start_idx < end_idx:
                self.assertTrue(torch.equal(embeddings, self.loss.embedding_memory[start_idx:end_idx]))
                self.assertTrue(torch.equal(labels, self.loss.label_memory[start_idx:end_idx]))
            else:
                correct_embeddings = torch.cat([self.loss.embedding_memory[start_idx:], self.loss.embedding_memory[:end_idx]], dim=0)
                correct_labels = torch.cat([self.loss.label_memory[start_idx:], self.loss.label_memory[:end_idx]], dim=0)
                self.assertTrue(torch.equal(embeddings, correct_embeddings))
                self.assertTrue(torch.equal(labels, correct_labels))

    def test_shift_indices_tuple(self):
        batch_size = 32
        pair_miner = PairMarginMiner(pos_margin=0, neg_margin=1, use_similarity=False)
        triplet_miner = TripletMarginMiner(margin=1)
        self.loss = CrossBatchMemory(loss=ContrastiveLoss(), embedding_size=self.embedding_size, memory_size=self.memory_size)
        for i in range(30):
            embeddings = torch.randn(batch_size, self.embedding_size)
            labels = torch.arange(batch_size)
            loss = self.loss(embeddings, labels)
            all_labels = torch.cat([labels, self.loss.label_memory], dim=0)

            indices_tuple = lmu.get_all_pairs_indices(labels, self.loss.label_memory)
            shifted = c_f.shift_indices_tuple(indices_tuple, batch_size)
            self.assertTrue(torch.equal(indices_tuple[0], shifted[0]))
            self.assertTrue(torch.equal(indices_tuple[2], shifted[2]))
            self.assertTrue(torch.equal(indices_tuple[1], shifted[1]-batch_size))
            self.assertTrue(torch.equal(indices_tuple[3], shifted[3]-batch_size))
            a1, p, a2, n = shifted
            self.assertTrue(not torch.any((all_labels[a1]-all_labels[p]).bool()))
            self.assertTrue(torch.all((all_labels[a2]-all_labels[n]).bool()))
            
            indices_tuple = pair_miner(embeddings, labels, self.loss.embedding_memory, self.loss.label_memory)
            shifted = c_f.shift_indices_tuple(indices_tuple, batch_size)
            self.assertTrue(torch.equal(indices_tuple[0], shifted[0]))
            self.assertTrue(torch.equal(indices_tuple[2], shifted[2]))
            self.assertTrue(torch.equal(indices_tuple[1], shifted[1]-batch_size))
            self.assertTrue(torch.equal(indices_tuple[3], shifted[3]-batch_size))
            a1, p, a2, n = shifted
            self.assertTrue(not torch.any((all_labels[a1]-all_labels[p]).bool()))
            self.assertTrue(torch.all((all_labels[a2]-all_labels[n]).bool()))

            indices_tuple = triplet_miner(embeddings, labels, self.loss.embedding_memory, self.loss.label_memory)
            shifted = c_f.shift_indices_tuple(indices_tuple, batch_size)
            self.assertTrue(torch.equal(indices_tuple[0], shifted[0]))
            self.assertTrue(torch.equal(indices_tuple[1], shifted[1]-batch_size))
            self.assertTrue(torch.equal(indices_tuple[2], shifted[2]-batch_size))
            a, p, n = shifted
            self.assertTrue(not torch.any((all_labels[a]-all_labels[p]).bool()))
            self.assertTrue(torch.all((all_labels[p]-all_labels[n]).bool()))


    def test_input_indices_tuple(self):
        batch_size = 32
        pair_miner = PairMarginMiner(pos_margin=0, neg_margin=1, use_similarity=False)
        triplet_miner = TripletMarginMiner(margin=1)
        self.loss = CrossBatchMemory(loss=ContrastiveLoss(), embedding_size=self.embedding_size, memory_size=self.memory_size)
        for i in range(30):
            embeddings = torch.randn(batch_size, self.embedding_size)
            labels = torch.arange(batch_size)
            self.loss(embeddings, labels)
            for curr_miner in [pair_miner, triplet_miner]:
                input_indices_tuple = curr_miner(embeddings, labels)
                all_labels = torch.cat([labels, self.loss.label_memory], dim=0)
                a1ii, pii, a2ii, nii = lmu.convert_to_pairs(input_indices_tuple, labels)
                a1i, pi, a2i, ni = lmu.get_all_pairs_indices(labels, self.loss.label_memory)
                a1, p, a2, n = self.loss.create_indices_tuple(batch_size, embeddings, labels, self.loss.embedding_memory, self.loss.label_memory, input_indices_tuple)
                self.assertTrue(not torch.any((all_labels[a1]-all_labels[p]).bool()))
                self.assertTrue(torch.all((all_labels[a2]-all_labels[n]).bool()))
                self.assertTrue(len(a1) == len(a1i)+len(a1ii))
                self.assertTrue(len(p) == len(pi)+len(pii))
                self.assertTrue(len(a2) == len(a2i)+len(a2ii))
                self.assertTrue(len(n) == len(ni)+len(nii))



if __name__ == '__main__':
    unittest.main()