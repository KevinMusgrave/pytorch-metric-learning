#! /usr/bin/env python3

from .. import miners
import torch
from ..utils import common_functions as c_f, loss_and_miner_utils as lmu

from . import train_with_classifier as twc
import copy

class DeepAdversarialMetricLearning(twc.TrainWithClassifier):
    def __init__(
        self,
        metric_alone_epochs=0,
        g_alone_epochs=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.original_loss_weights = copy.deepcopy(self.loss_weights)
        self.metric_alone_epochs = metric_alone_epochs
        self.g_alone_epochs = g_alone_epochs
        self.loss_funcs["G_neg_adv"].maybe_modify_loss = lambda x: x * -1

    def custom_setup(self):
        synth_packaged_as_triplets = miners.EmbeddingsAlreadyPackagedAsTriplets(
            normalize_embeddings=False)
        self.mining_funcs["synth_packaged_as_triplets"] = synth_packaged_as_triplets

    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        penultimate_embeddings = self.get_trunk_output(data)

        if self.do_metric:
            authentic_final_embeddings = self.get_final_embeddings(penultimate_embeddings)
            authentic_final_embeddings, labels = self.post_processor(authentic_final_embeddings, labels)
            indices_tuple = self.maybe_mine_embeddings(authentic_final_embeddings, labels)
            self.losses["metric_loss"] = self.loss_funcs["metric_loss"](
                authentic_final_embeddings, labels, indices_tuple
            )
            logits = self.maybe_get_logits(authentic_final_embeddings)
            self.losses["classifier_loss"] = self.maybe_get_classifier_loss(logits, labels)

        if self.do_adv:
            self.calculate_synth_loss(penultimate_embeddings, labels)

    def loss_names(self):
        return ["metric_loss", "classifier_loss", "synth_loss", "G_neg_hard", "G_neg_reg", "G_neg_adv"]

    def update_loss_weights(self):
        self.do_metric_alone = self.epoch <= self.metric_alone_epochs
        self.do_adv_alone = self.metric_alone_epochs < self.epoch <= self.g_alone_epochs
        self.do_both = not self.do_adv_alone and not self.do_metric_alone
        self.do_adv = self.do_adv_alone or self.do_both
        self.do_metric = self.do_metric_alone or self.do_both

        non_zero_weight_list = []
        if self.do_adv:
            non_zero_weight_list += ["G_neg_hard", "G_neg_reg", "G_neg_adv"]
        if self.do_metric:
            non_zero_weight_list += ["metric_loss", "classifier_loss"]
        if self.do_both:
            non_zero_weight_list += ["synth_loss"]

        for k in self.loss_weights.keys():
            if k in non_zero_weight_list:
                self.loss_weights[k] = self.original_loss_weights[k]
            else:
                self.loss_weights[k] = 0

        self.maybe_exclude_networks_from_gradient()

    def maybe_exclude_networks_from_gradient(self):
        self.set_to_train()
        self.maybe_freeze_trunk_batchnorm()
        if self.do_adv_alone:
            no_grad_list = ["trunk", "classifier"]
        elif self.do_metric_alone:
            no_grad_list = ["G_neg_model"]
        else:
            no_grad_list = []
        for k in self.models.keys():
            if k in no_grad_list:
                c_f.set_requires_grad(self.models[k], requires_grad=False)
                self.models[k].eval()
            else:
                c_f.set_requires_grad(self.models[k], requires_grad=True)


    def step_optimizers(self):
        step_list = []
        if self.do_metric:
            step_list += ["trunk_optimizer", "embedder_optimizer", "classifier_optimizer"]
        if self.do_adv:
            step_list += ["G_neg_model_optimizer"]
        for k in self.optimizers.keys():
            if k in step_list:
                self.optimizers[k].step()

    def calculate_synth_loss(self, penultimate_embeddings, labels):
        a_indices, p_indices, n_indices = lmu.get_random_triplet_indices(labels, t_per_anchor=10)
        real_anchors = penultimate_embeddings[a_indices]
        real_positives = penultimate_embeddings[p_indices]
        real_negatives = penultimate_embeddings[n_indices]
        penultimate_embeddings_cat = torch.cat([real_anchors, real_positives, real_negatives], dim=1)
        synthetic_negatives = c_f.pass_data_to_model(
            self.models["G_neg_model"], penultimate_embeddings_cat, self.data_device
        )
        penultimate_embeddings_with_negative_synth = c_f.unslice_by_n(
            [real_anchors, real_positives, synthetic_negatives]
        )
        final_embeddings = self.get_final_embeddings(penultimate_embeddings_with_negative_synth)

        labels = torch.tensor(
            [
                val
                for tup in zip(
                    *[labels[a_indices], labels[p_indices], labels[n_indices]]
                )
                for val in tup
            ]
        )

        final_embeddings, labels = self.post_processor(final_embeddings, labels)

        indices_tuple = self.mining_funcs["synth_packaged_as_triplets"](final_embeddings, labels)

        if self.do_both:
            self.losses["synth_loss"] = self.loss_funcs["synth_loss"](
                final_embeddings, labels, indices_tuple
            )

        self.losses["G_neg_adv"] = self.loss_funcs["G_neg_adv"](
            final_embeddings, labels, indices_tuple
        )
        self.losses["G_neg_hard"] = torch.nn.functional.mse_loss(
            torch.nn.functional.normalize(synthetic_negatives, p=2, dim=1),
            torch.nn.functional.normalize(real_anchors, p=2, dim=1),
        )
        self.losses["G_neg_reg"] = torch.nn.functional.mse_loss(
            torch.nn.functional.normalize(synthetic_negatives, p=2, dim=1),
            torch.nn.functional.normalize(real_negatives, p=2, dim=1),
        )
