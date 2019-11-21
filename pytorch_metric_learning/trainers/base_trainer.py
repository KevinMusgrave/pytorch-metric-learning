#! /usr/bin/env python3

import torch
from ..utils import common_functions as c_f, loss_tracker as l_t
import tqdm
import logging

class BaseTrainer:
    def __init__(
        self,
        models,
        optimizers,
        batch_size,
        loss_funcs,
        mining_funcs,
        num_epochs,
        iterations_per_epoch,
        dataset,
        data_device=None,
        loss_weights=None,
        label_mapper=None,
        sampler=None,
        collate_fn=None,
        record_keeper=None,
        lr_schedulers=None,
        gradient_clippers=None,
        freeze_trunk_batchnorm=False,
        label_hierarchy_level=0,
        dataloader_num_workers=32,
        post_processor=None,
        start_epoch=1,
        possible_data_keys=None
    ):
        self.models = models
        self.optimizers = optimizers
        self.batch_size = batch_size
        self.loss_funcs = loss_funcs
        self.mining_funcs = mining_funcs
        self.label_mapper = label_mapper
        self.num_epochs = num_epochs
        self.iterations_per_epoch = iterations_per_epoch
        self.dataset = dataset
        self.data_device = data_device
        self.sampler = sampler
        self.collate_fn = collate_fn
        self.record_keeper = record_keeper
        self.lr_schedulers = lr_schedulers
        self.gradient_clippers = gradient_clippers
        self.freeze_trunk_batchnorm = freeze_trunk_batchnorm
        self.label_hierarchy_level = label_hierarchy_level
        self.dataloader_num_workers = dataloader_num_workers
        self.post_processor = post_processor
        self.epoch = start_epoch
        self.loss_weights = loss_weights
        self.possible_data_keys = possible_data_keys
        self.custom_setup()
        self.initialize_data_device()
        self.initialize_label_mapper()
        self.initialize_post_processor()
        self.initialize_loss_tracker()
        self.initialize_dataloader()
        self.initialize_loss_weights()
        self.initialize_possible_data_keys()

    def custom_setup(self):
        pass

    def calculate_loss(self):
        raise NotImplementedError

    def loss_names(self):
        raise NotImplementedError

    def update_loss_weights(self):
        pass

    def train(self):
        self.set_to_train()
        while self.epoch <= self.num_epochs:
            logging.info("TRAINING EPOCH %d" % self.epoch)
            pbar = tqdm.tqdm(range(self.iterations_per_epoch))
            for self.iteration in pbar:
                self.forward_and_backward()
                pbar.set_description("total_loss=%.5f" % self.losses["total_loss"])
            self.step_lr_schedulers()
            self.update_records(end_of_epoch=True)
            self.epoch += 1

    def initialize_dataloader(self):
        self.dataloader = c_f.get_dataloader(
            self.dataset,
            self.batch_size,
            self.sampler,
            self.dataloader_num_workers,
            self.collate_fn,
        )
        self.dataloader_iter = iter(self.dataloader)

    def forward_and_backward(self):
        self.zero_losses()
        self.zero_grad()
        self.update_loss_weights()
        self.calculate_loss(self.get_batch())
        self.loss_tracker.update(self.loss_weights)
        self.update_records()
        self.backward()
        self.clip_gradients()
        self.step_optimizers()

    def zero_losses(self):
        for k in self.losses.keys():
            self.losses[k] = 0

    def zero_grad(self):
        for v in self.models.values():
            v.zero_grad()
        for v in self.optimizers.values():
            v.zero_grad()

    def get_batch(self):
        self.dataloader_iter, curr_batch = c_f.try_next_on_generator(
            self.dataloader_iter, self.dataloader)
        curr_batch["label"] = c_f.process_label(
            curr_batch["label"], self.label_hierarchy_level, self.label_mapper
        )
        curr_batch = self.maybe_do_pre_gradient_mining(curr_batch)
        return c_f.try_keys(curr_batch, self.possible_data_keys), curr_batch["label"]

    def compute_embeddings(self, data, labels):
        trunk_output = self.get_trunk_output(data)
        embeddings = self.get_final_embeddings(trunk_output)
        embeddings, labels = self.post_processor(embeddings, labels)
        return embeddings, labels

    def get_final_embeddings(self, base_output):
        return self.models["embedder"](base_output)

    def get_trunk_output(self, data):
        return c_f.pass_data_to_model(self.models["trunk"], data, self.data_device)

    def maybe_mine_embeddings(self, embeddings, labels):
        if "post_gradient_miner" in self.mining_funcs:
            return self.mining_funcs["post_gradient_miner"](embeddings, labels)
        return None

    def maybe_do_pre_gradient_mining(self, curr_batch):
        if "pre_gradient_miner" in self.mining_funcs:
            with torch.no_grad():
                self.set_to_eval()
                data = c_f.try_keys(curr_batch, self.possible_data_keys)
                labels = curr_batch["label"]
                embeddings, labels = self.compute_embeddings(data, labels)
                idx = self.mining_funcs["pre_gradient_miner"](embeddings, labels)
                self.set_to_train()
            curr_batch = {self.possible_data_keys[0]: data[idx], "label": labels[idx]}
        return curr_batch

    def backward(self):
        if self.losses["total_loss"] > 0.0:
            self.losses["total_loss"].backward()

    def get_global_iteration(self):
        return self.iteration + self.iterations_per_epoch * (self.epoch - 1)

    def step_lr_schedulers(self):
        if self.lr_schedulers is not None:
            for v in self.lr_schedulers.values():
                v.step()

    def step_optimizers(self):
        for v in self.optimizers.values():
            v.step()

    def clip_gradients(self):
        if self.gradient_clippers is not None:
            for v in self.gradient_clippers.values():
                v()

    def maybe_freeze_trunk_batchnorm(self):
        if self.freeze_trunk_batchnorm:
            self.models["trunk"].apply(c_f.set_layers_to_eval("BatchNorm"))

    def initialize_post_processor(self):
        if self.post_processor is None:
            self.post_processor = lambda embeddings, labels: (embeddings, labels)

    def initialize_data_device(self):
        if self.data_device is None:
            self.data_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize_label_mapper(self):
        if self.label_mapper is None:
            self.label_mapper = lambda label, hierarchy_level: label

    def initialize_loss_tracker(self):
        self.loss_tracker = l_t.LossTracker(self.loss_names())
        self.losses = self.loss_tracker.losses

    def initialize_possible_data_keys(self):
        if self.possible_data_keys is None:
            self.possible_data_keys = ["data", "image"]

    def set_to_train(self):
        for k, v in self.models.items():
            self.models[k] = v.train()
        self.maybe_freeze_trunk_batchnorm()

    def set_to_eval(self):
        for k, v in self.models.items():
            self.models[k] = v.eval()

    def initialize_loss_weights(self):
        if self.loss_weights is None:
            self.loss_weights = {k: 1 for k in self.loss_names()}

    def update_records(self, end_of_epoch=False):
        if self.record_keeper is not None:
            if end_of_epoch:
                self.record_keeper.maybe_add_custom_figures_to_tensorboard(self.get_global_iteration())
            else:
               for record, kwargs in self.record_these():
                  self.record_keeper.update_records(record, self.get_global_iteration(), **kwargs)

    def record_these(self):
        return [[self.loss_tracker.losses, {"input_group_name_for_non_objects": "loss_histories"}],
                [self.loss_tracker.loss_weights, {"input_group_name_for_non_objects": "loss_weights"}],
                [self.loss_funcs, {}],
                [self.mining_funcs, {}],
                [self.models, {}],
                [self.optimizers, {"custom_attr_func": lambda x: {"lr": x.param_groups[0]["lr"]}}]]
