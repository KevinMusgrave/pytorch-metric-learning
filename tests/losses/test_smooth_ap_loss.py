import unittest

import torch
import torch.nn.functional as F

from pytorch_metric_learning.losses import SmoothAPLoss

from .. import TEST_DEVICE, TEST_DTYPES

TEMPERATURE = 0.01
BATCH_SIZE = 60
NUM_ID = 6
FEAT_DIMS = 256
TEST_SEEDS = [42, 1234, 5642, 9999, 3459]


def sigmoid(tensor, temp=1.0):
    exponent = -tensor / temp
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


def compute_aff(x):
    return torch.mm(x, x.t())


class SmoothAP(torch.nn.Module):
    def __init__(self, anneal, batch_size, num_id, feat_dims):
        super().__init__()

        assert batch_size % num_id == 0

        self.anneal = anneal
        self.batch_size = batch_size
        self.num_id = num_id
        self.feat_dims = feat_dims

    def forward(self, preds):
        mask = 1.0 - torch.eye(self.batch_size)
        mask = mask.unsqueeze(dim=0).repeat(self.batch_size, 1, 1)
        sim_all = compute_aff(preds)
        sim_all_repeat = sim_all.unsqueeze(dim=1).repeat(1, self.batch_size, 1)
        sim_diff = sim_all_repeat - sim_all_repeat.permute(0, 2, 1)
        sim_sg = sigmoid(sim_diff, temp=self.anneal) * mask.to(TEST_DEVICE)
        sim_all_rk = torch.sum(sim_sg, dim=-1) + 1

        xs = preds.view(self.num_id, int(self.batch_size / self.num_id), self.feat_dims)
        pos_mask = 1.0 - torch.eye(int(self.batch_size / self.num_id))
        pos_mask = (
            pos_mask.unsqueeze(dim=0)
            .unsqueeze(dim=0)
            .repeat(self.num_id, int(self.batch_size / self.num_id), 1, 1)
        )

        sim_pos = torch.bmm(xs, xs.permute(0, 2, 1))
        sim_pos_repeat = sim_pos.unsqueeze(dim=2).repeat(
            1, 1, int(self.batch_size / self.num_id), 1
        )
        sim_pos_diff = sim_pos_repeat - sim_pos_repeat.permute(0, 1, 3, 2)
        sim_pos_sg = sigmoid(sim_pos_diff, temp=self.anneal) * pos_mask.to(TEST_DEVICE)
        sim_pos_rk = torch.sum(sim_pos_sg, dim=-1) + 1

        ap = torch.zeros(1).to(TEST_DEVICE)
        group = int(self.batch_size / self.num_id)
        for ind in range(self.num_id):
            pos_divide = torch.sum(
                sim_pos_rk[ind]
                / (
                    sim_all_rk[
                        (ind * group) : ((ind + 1) * group),
                        (ind * group) : ((ind + 1) * group),
                    ]
                )
            )
            ap = ap + ((pos_divide / group) / self.batch_size)

        return 1 - ap


def contiguous_equal_size_class_labels(num_classes, batch_size):
    group = batch_size // num_classes
    return torch.arange(num_classes).repeat_interleave(group)


class TestSmoothAPLoss(unittest.TestCase):
    def test_smooth_ap_loss(self):
        for dtype in TEST_DTYPES:
            for seed in TEST_SEEDS:
                torch.manual_seed(seed)
                loss = SmoothAP(TEMPERATURE, BATCH_SIZE, NUM_ID, FEAT_DIMS)
                rand_tensor = (
                    torch.randn(BATCH_SIZE, FEAT_DIMS, requires_grad=True)
                    .to(TEST_DEVICE)
                    .to(dtype)
                )
                normalized_model_output = F.normalize(rand_tensor, p=2.0, dim=-1)
                output = loss(normalized_model_output)

                loss2 = SmoothAPLoss(temperature=TEMPERATURE)
                labels = contiguous_equal_size_class_labels(NUM_ID, BATCH_SIZE)
                output2 = loss2.forward(rand_tensor, labels)
                self.assertEqual(output2.dtype, dtype)
                rtol, atol = (1e-2, 1e-4) if dtype == torch.float16 else (1e-5, 1e-8)
                self.assertTrue(
                    torch.isclose(output.double(), output2.double(), rtol, atol)
                )


def sigmoid_rank_step(sims, query, item, other, temperature):
    return torch.sigmoid((sims[query, other] - sims[query, item]) / temperature)


def direct_smooth_ap(embeddings, labels, temperature):
    x = F.normalize(embeddings.double(), dim=1)
    sims = x @ x.T
    batch = len(labels)
    precisions = []
    for q in range(batch):
        positives = [i for i in range(batch) if labels[i] == labels[q]]
        total = sims.new_zeros(())
        for i in positives:
            all_rank = 1 + sum(
                sigmoid_rank_step(sims, q, i, j, temperature)
                for j in range(batch)
                if j != i
            )
            positive_rank = 1 + sum(
                sigmoid_rank_step(sims, q, i, j, temperature)
                for j in positives
                if j != i
            )
            total = total + positive_rank / all_rank
        precisions.append(total / len(positives))
    return (1 - torch.stack(precisions)).mean()


class TestSmoothAPLabelSets(unittest.TestCase):
    def test_arbitrary_label_order_and_class_sizes(self):
        torch.manual_seed(3)
        label_sets = (
            [0, 0, 1, 1, 2, 2],
            [0, 1, 0, 1],
            [2, 0, 2, 1, 1, 0, 0],
            [5, 5, 5, 9],
        )
        for labels in label_sets:
            for dtype in TEST_DTYPES:
                with self.subTest(labels=labels, dtype=dtype):
                    labels_tensor = torch.tensor(labels, device=TEST_DEVICE)
                    embeddings = torch.randn(
                        len(labels),
                        8,
                        device=TEST_DEVICE,
                        dtype=dtype,
                        requires_grad=True,
                    )
                    actual = SmoothAPLoss(temperature=0.1)(embeddings, labels_tensor)
                    expected = direct_smooth_ap(embeddings, labels_tensor, 0.1)
                    self.assertEqual(actual.dtype, dtype)
                    rtol, atol = {
                        torch.float16: (1e-2, 1e-4),
                        torch.float32: (1e-4, 1e-5),
                    }.get(dtype, (None, None))
                    torch.testing.assert_close(
                        actual.double(), expected, rtol=rtol, atol=atol
                    )
                    torch.testing.assert_close(
                        torch.autograd.grad(actual, embeddings)[0].double(),
                        torch.autograd.grad(expected, embeddings)[0].double(),
                        rtol=rtol,
                        atol=atol,
                    )
