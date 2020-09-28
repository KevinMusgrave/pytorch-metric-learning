import unittest
from .. import TEST_DTYPES, TEST_DEVICE
import torch
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity


class TestTripletMarginMiner(unittest.TestCase):
    def test_triplet_margin_miner(self):
        for dtype in TEST_DTYPES:
            for distance in [LpDistance(), CosineSimilarity()]:
                embedding_angles = torch.arange(0, 16)
                embeddings = torch.tensor(
                    [c_f.angle_to_coord(a) for a in embedding_angles],
                    requires_grad=True,
                    dtype=dtype,
                ).to(
                    TEST_DEVICE
                )  # 2D embeddings
                labels = torch.randint(low=0, high=2, size=(16,))
                mat = distance(embeddings)
                triplets = []
                for i in range(len(embeddings)):
                    anchor_label = labels[i]
                    for j in range(len(embeddings)):
                        if j == i:
                            continue
                        positive_label = labels[j]
                        if positive_label == anchor_label:
                            ap_dist = mat[i, j]
                            for k in range(len(embeddings)):
                                if k == j or k == i:
                                    continue
                                negative_label = labels[k]
                                if negative_label != positive_label:
                                    an_dist = mat[i, k]
                                    if distance.is_inverted:
                                        triplets.append((i, j, k, ap_dist - an_dist))
                                    else:
                                        triplets.append((i, j, k, an_dist - ap_dist))

                for margin_int in range(-1, 11):
                    margin = float(margin_int) * 0.05
                    minerA = TripletMarginMiner(
                        margin, type_of_triplets="all", distance=distance
                    )
                    minerB = TripletMarginMiner(
                        margin, type_of_triplets="hard", distance=distance
                    )
                    minerC = TripletMarginMiner(
                        margin, type_of_triplets="semihard", distance=distance
                    )
                    minerD = TripletMarginMiner(
                        margin, type_of_triplets="easy", distance=distance
                    )

                    correctA, correctB, correctC, correctD = [], [], [], []
                    for i, j, k, distance_diff in triplets:
                        if distance_diff > margin:
                            correctD.append((i, j, k))
                        else:
                            correctA.append((i, j, k))
                            if distance_diff > 0:
                                correctC.append((i, j, k))
                            if distance_diff <= 0:
                                correctB.append((i, j, k))

                    for correct, miner in [
                        (correctA, minerA),
                        (correctB, minerB),
                        (correctC, minerC),
                        (correctD, minerD),
                    ]:
                        correct_triplets = set(correct)
                        a1, p1, n1 = miner(embeddings, labels)
                        mined_triplets = set(
                            [
                                (a.item(), p.item(), n.item())
                                for a, p, n in zip(a1, p1, n1)
                            ]
                        )
                        self.assertTrue(mined_triplets == correct_triplets)

    def test_empty_output(self):
        miner = TripletMarginMiner(0.5, type_of_triplets="all")
        for dtype in TEST_DTYPES:
            batch_size = 32
            embeddings = torch.randn(batch_size, 64).type(dtype).to(TEST_DEVICE)
            labels = torch.arange(batch_size)
            a, p, n = miner(embeddings, labels)
            self.assertTrue(len(a) == 0)
            self.assertTrue(len(p) == 0)
            self.assertTrue(len(n) == 0)
