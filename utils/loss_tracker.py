#! /usr/bin/env python3


class LossTracker:
    def __init__(self, loss_names):
        if "total_loss" not in loss_names:
            loss_names.append("total_loss")
        self.losses = {key: 0 for key in loss_names}
        self.loss_weights = {key: 1 for key in loss_names}

    def weight_the_losses(self, exclude_loss=("total_loss")):
        for k, _ in self.losses.items():
            if k not in exclude_loss:
                self.losses[k] *= self.loss_weights[k]

    def get_total_loss(self, exclude_loss=("total_loss")):
        self.losses["total_loss"] = 0
        for k, v in self.losses.items():
            if k not in exclude_loss:
                self.losses["total_loss"] += v

    def set_loss_weights(self, loss_weight_dict):
        for k, _ in self.losses.items():
            if k in loss_weight_dict:
                w = loss_weight_dict[k]
            else:
                w = 1.0
            self.loss_weights[k] = w

    def update(self, loss_weight_dict):
        self.set_loss_weights(loss_weight_dict)
        self.weight_the_losses()
        self.get_total_loss()
