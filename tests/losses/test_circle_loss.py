import unittest

import torch

from pytorch_metric_learning.losses import CircleLoss
from ..zzz_testing_utils.testing_utils import angle_to_coord

from .. import TEST_DEVICE, TEST_DTYPES


class TestCircleLoss(unittest.TestCase):
    def test_circle_loss(self):
        margin, gamma = 0.25, 256
        Op, On = 1 + margin, -margin
        delta_p, delta_n = 1 - margin, margin
        loss_func = CircleLoss(m=margin, gamma=gamma)

        for dtype in TEST_DTYPES:
            embedding_angles = [0, 20, 40, 60, 80]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.LongTensor([0, 0, 1, 1, 2])

            loss = loss_func(embeddings, labels)
            loss.backward()

            pos_pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]
            neg_pairs = [
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 0),
                (2, 1),
                (2, 4),
                (3, 0),
                (3, 1),
                (3, 4),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 3),
            ]

            normalized_embeddings = torch.nn.functional.normalize(embeddings)
            correct_total = 0
            for i in range(len(normalized_embeddings)):
                pos_logits = []
                neg_logits = []
                for a, p in pos_pairs:
                    if a == i:
                        anchor, positive = (
                            normalized_embeddings[a],
                            normalized_embeddings[p],
                        )
                        ap_sim = torch.matmul(anchor, positive)
                        logit_p = -gamma * torch.relu(Op - ap_sim) * (ap_sim - delta_p)
                        pos_logits.append(logit_p.unsqueeze(0))

                for a, n in neg_pairs:
                    if a == i:
                        anchor, negative = (
                            normalized_embeddings[a],
                            normalized_embeddings[n],
                        )
                        an_sim = torch.matmul(anchor, negative)
                        logit_n = gamma * torch.relu(an_sim - On) * (an_sim - delta_n)
                        neg_logits.append(logit_n.unsqueeze(0))

                if len(pos_logits) == 0 or len(neg_logits) == 0:
                    pass
                else:
                    pos_logits = (
                        torch.cat(pos_logits, dim=0)
                        if len(pos_logits) > 1
                        else pos_logits[0]
                    )
                    neg_logits = (
                        torch.cat(neg_logits, dim=0)
                        if len(neg_logits) > 1
                        else neg_logits[0]
                    )
                    correct_total += torch.nn.functional.softplus(
                        torch.logsumexp(pos_logits, dim=0)
                        + torch.logsumexp(neg_logits, dim=0)
                    )

            correct_total /= 4  # only 4 of the embeddings have both pos and neg pairs
            self.assertTrue(torch.isclose(loss, correct_total))

    def test_with_no_valid_pairs(self):
        margin, gamma = 0.4, 80
        loss_func = CircleLoss(m=margin, gamma=gamma)
        for dtype in TEST_DTYPES:
            embedding_angles = [0]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.LongTensor([0])
            loss = loss_func(embeddings, labels)
            loss.backward()
            self.assertEqual(loss, 0)

    def test_overflow(self):
        margin, gamma = 0.4, 300
        loss_func = CircleLoss(m=margin, gamma=gamma)
        for dtype in TEST_DTYPES:
            embedding_angles = torch.arange(0, 180)
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.randint(low=0, high=10, size=(180,))
            loss = loss_func(embeddings, labels)
            loss.backward()
            self.assertTrue(not torch.isnan(loss) and not torch.isinf(loss))
            self.assertTrue(loss > 0)


if __name__ == "__main__":
    unittest.main()
