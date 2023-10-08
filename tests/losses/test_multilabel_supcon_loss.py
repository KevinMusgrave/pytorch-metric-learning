import unittest

import torch
import numpy as np

from pytorch_metric_learning.losses import (
    MultiSupConLoss,
    CrossBatchMemory4MultiLabel
)

from ..zzz_testing_utils.testing_utils import angle_to_coord

from .. import TEST_DEVICE, TEST_DTYPES
class TestMultiSupConLoss(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.n_cls = 3
        self.n_samples = 4
        self.n_dim = 3
        self.n_batchs = 10
        self.xbm_max_size = 1024

        # multi_supcon
        self.loss_func = MultiSupConLoss(
                        num_classes=self.n_cls,
                        temperature=0.07,
                        threshold=0.3)

        # xbm
        self.xbm_loss_func = CrossBatchMemory4MultiLabel(
                        self.loss_func, 
                        self.n_dim, 
                        memory_size=self.xbm_max_size)
        # test cases
        self.embeddings = torch.tensor([[0.1, 0.3, 0.1],
                                [0.23, -0.2, -0.1],
                                [0.1, -0.16, 0.1],
                                [0.13, -0.13, 0.2]])
        self.labels = torch.tensor([[1,0,1], [1,0,0], [0,1,1], [0,1,0]])

        # the gt values are obtained by running the code 
        # (https://github.com/WolodjaZ/MultiSupContrast/blob/main/losses.py)
        
        # multi_supcon test cases
        self.test_multisupcon_val_gt = {
            torch.float16: 3.2836,
            torch.float32: 3.2874,
            torch.float64: 3.2874,
        }
        # xbm test cases
        self.test_xbm_multisupcon_val_gt = {
            torch.float16: [3.2836, 4.3792, 4.4588, 4.5741, 4.6831, 4.7809, 4.8682, 4.9465, 5.0174, 5.0819],
            torch.float32: [3.2874, 4.3779, 4.4577, 4.5730, 4.6820, 4.7798, 4.8671, 4.9455, 5.0163, 5.0808],
            torch.float64: [3.2874, 4.3779, 4.4577, 4.5730, 4.6820, 4.7798, 4.8671, 4.9455, 5.0163, 5.0808,]
        }


    def test_multisupcon_val(self):
        for dtype in TEST_DTYPES:
            for device in ["cpu", "cuda"]:
                # skip float16 on cpu
                if device == "cpu" and dtype == torch.float16:
                    continue
                embedding = self.embeddings.to(device).to(dtype)
                label = self.labels.to(device).to(dtype)
                loss = self.loss_func(embedding, label)
                loss = loss.to("cpu")
                self.assertTrue(np.isclose(
                    loss.item(), 
                    self.test_multisupcon_val_gt[dtype], 
                    atol=1e-2 if dtype == torch.float16 else 1e-4))


    def test_xbm_multisupcon_val(self):
        # test xbm with scatter labels
        for dtype in TEST_DTYPES:
            for device in ["cpu", "cuda"]:
                # skip float16 on cpu
                if device == "cpu" and dtype == torch.float16:
                    continue
                self.xbm_loss_func.reset_queue()
                for b in range(self.n_batchs):            
                    embedding = self.embeddings.to(device).to(dtype)
                    label = self.labels.to(device).to(dtype)
                    loss = self.xbm_loss_func(embedding, label)
                    loss = loss.to("cpu")
                    print(loss, self.test_xbm_multisupcon_val_gt[dtype][b], dtype)
                    self.assertTrue(np.isclose(
                        loss.item(), 
                        self.test_xbm_multisupcon_val_gt[dtype][b], 
                        atol=1e-2 if dtype == torch.float16 else 1e-4))
                    
    def test_with_no_valid_pairs(self):
        for dtype in TEST_DTYPES:
            embedding_angles = [0]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.LongTensor([[0]])
            loss = self.loss_func(embeddings, labels)
            loss.backward()
            self.assertEqual(loss, 0)

    def test_(self):
        for dtype in TEST_DTYPES:
            embedding_angles = [0]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.LongTensor([[0]])
            loss = self.loss_func(embeddings, labels)
            loss.backward()
            self.assertEqual(loss, 0)

    
    def test_backward(self):
        for dtype in TEST_DTYPES:
            embedding_angles = list(range(0, 180, 20))[:4]
            embeddings = torch.tensor(
                [angle_to_coord(a) for a in embedding_angles],
                requires_grad=True,
                dtype=dtype,
            ).to(
                TEST_DEVICE
            )  # 2D embeddings
            labels = torch.LongTensor([[0, 0, 1, 0, 1, 0, 0],
                                       [1, 0, 1, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0, 0],
                                       [0, 0, 1, 0, 1, 0, 1]]).to(TEST_DEVICE)

            loss = self.loss_func(embeddings, labels)
            loss.backward()