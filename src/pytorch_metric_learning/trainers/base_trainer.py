import logging

import torch
import tqdm

from ..utils import common_functions as c_f
from ..utils import loss_tracker as l_t


class BaseTrainer:
    def __init__(
        self,
        models,
        optimizers,
        batch_size,
        loss_funcs,
        mining_funcs,
        dataset,
        iterations_per_epoch=None,
        data_device=None,
        dtype=None,
        loss_weights=None,
        sampler=None,
        collate_fn=None,
        lr_schedulers=None,
        gradient_clippers=None,
        freeze_these=(),
        freeze_trunk_batchnorm=False,
        label_hierarchy_level=0,
        dataloader_num_workers=32,
        data_and_label_getter=None,
        dataset_labels=None,
        set_min_label_to_zero=False,
        end_of_iteration_hook=None,
        end_of_epoch_hook=None,
    ):
        self.models = models
        self.optimizers = optimizers
        self.batch_size = batch_size
        self.loss_funcs = loss_funcs
        self.mining_funcs = mining_funcs
        self.dataset = dataset
        self.iterations_per_epoch = iterations_per_epoch
        self.data_device = data_device
        self.dtype = dtype
        self.sampler = sampler
        self.collate_fn = collate_fn
        self.lr_schedulers = lr_schedulers
        self.gradient_clippers = gradient_clippers
        self.freeze_these = freeze_these
        self.freeze_trunk_batchnorm = freeze_trunk_batchnorm
        self.label_hierarchy_level = label_hierarchy_level
        self.dataloader_num_workers = dataloader_num_workers
        self.loss_weights = loss_weights
        self.data_and_label_getter = data_and_label_getter
        self.dataset_labels = dataset_labels
        self.set_min_label_to_zero = set_min_label_to_zero
        self.end_of_iteration_hook = end_of_iteration_hook
        self.end_of_epoch_hook = end_of_epoch_hook
        self.loss_names = list(self.loss_funcs.keys())
        self.custom_setup()
        self.verify_dict_keys()
        self.initialize_models()
        self.initialize_data_device()
        self.initialize_label_mapper()
        self.initialize_loss_tracker()
        self.initialize_loss_weights()
        self.initialize_data_and_label_getter()
        self.initialize_hooks()
        self.initialize_lr_schedulers()

    def custom_setup(self):
        pass

    def calculate_loss(self):
        raise NotImplementedError

    def update_loss_weights(self):
        pass

    def train(self, start_epoch=1, num_epochs=1):
        self.initialize_dataloader()
        for self.epoch in range(start_epoch, num_epochs + 1):
            self.set_to_train()
            logging.info("TRAINING EPOCH %d" % self.epoch)
            pbar = tqdm.tqdm(range(self.iterations_per_epoch))
            for self.iteration in pbar:
                self.forward_and_backward()
                self.end_of_iteration_hook(self)
                pbar.set_description("total_loss=%.5f" % self.losses["total_loss"])
                self.step_lr_schedulers(end_of_epoch=False)
            self.step_lr_schedulers(end_of_epoch=True)
            self.zero_losses()
            if self.end_of_epoch_hook(self) is False:
                break

    def initialize_dataloader(self):
        logging.info("Initializing dataloader")
        self.dataloader = c_f.get_train_dataloader(
            self.dataset,
            self.batch_size,
            self.sampler,
            self.dataloader_num_workers,
            self.collate_fn,
        )
        if not self.iterations_per_epoch:
            self.iterations_per_epoch = len(self.dataloader)
        logging.info("Initializing dataloader iterator")
        self.dataloader_iter = iter(self.dataloader)
        logging.info("Done creating dataloader iterator")

    def forward_and_backward(self):
        self.zero_losses()
        self.zero_grad()
        self.update_loss_weights()
        self.calculate_loss(self.get_batch())
        self.loss_tracker.update(self.loss_weights)
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
            self.dataloader_iter, self.dataloader
        )
        data, labels = self.data_and_label_getter(curr_batch)
        labels = c_f.process_label(
            labels, self.label_hierarchy_level, self.label_mapper
        )
        return self.maybe_do_batch_mining(data, labels)

    def compute_embeddings(self, data):
        trunk_output = self.get_trunk_output(data)
        embeddings = self.get_final_embeddings(trunk_output)
        return embeddings

    def get_final_embeddings(self, base_output):
        return self.models["embedder"](base_output)

    def get_trunk_output(self, data):
        data = data.to(self.data_device)
        if self.dtype is not None:
            data = data.type(self.dtype)
        return self.models["trunk"](data)

    def maybe_mine_embeddings(self, embeddings, labels):
        if "tuple_miner" in self.mining_funcs:
            return self.mining_funcs["tuple_miner"](embeddings, labels)
        return None

    def maybe_do_batch_mining(self, data, labels):
        if "subset_batch_miner" in self.mining_funcs:
            with torch.no_grad():
                self.set_to_eval()
                embeddings = self.compute_embeddings(data)
                idx = self.mining_funcs["subset_batch_miner"](embeddings, labels)
                self.set_to_train()
                data, labels = data[idx], labels[idx]
        return data, labels

    def backward(self):
        self.losses["total_loss"].backward()

    def get_global_iteration(self):
        return self.iteration + self.iterations_per_epoch * (self.epoch - 1)

    def step_lr_schedulers(self, end_of_epoch=False):
        if self.lr_schedulers is not None:
            for k, v in self.lr_schedulers.items():
                if end_of_epoch and k.endswith(
                    self.allowed_lr_scheduler_key_suffixes["epoch"]
                ):
                    v.step()
                elif not end_of_epoch and k.endswith(
                    self.allowed_lr_scheduler_key_suffixes["iteration"]
                ):
                    v.step()

    def step_lr_plateau_schedulers(self, validation_info):
        if self.lr_schedulers is not None:
            for k, v in self.lr_schedulers.items():
                if k.endswith(self.allowed_lr_scheduler_key_suffixes["plateau"]):
                    v.step(validation_info)

    def step_optimizers(self):
        for k, v in self.optimizers.items():
            if c_f.regex_replace("_optimizer$", "", k) not in self.freeze_these:
                v.step()

    def clip_gradients(self):
        if self.gradient_clippers is not None:
            for v in self.gradient_clippers.values():
                v()

    def maybe_freeze_trunk_batchnorm(self):
        if self.freeze_trunk_batchnorm:
            self.models["trunk"].apply(c_f.set_layers_to_eval("BatchNorm"))

    def initialize_data_device(self):
        if self.data_device is None:
            self.data_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

    def initialize_label_mapper(self):
        self.label_mapper = c_f.LabelMapper(
            self.set_min_label_to_zero, self.dataset_labels
        ).map

    def initialize_loss_tracker(self):
        self.loss_tracker = l_t.LossTracker(self.loss_names)
        self.losses = self.loss_tracker.losses

    def initialize_data_and_label_getter(self):
        if self.data_and_label_getter is None:
            self.data_and_label_getter = c_f.return_input

    def trainable_attributes(self):
        return [self.models, self.loss_funcs]

    def set_to_train(self):
        for T in self.trainable_attributes():
            for k, v in T.items():
                if k in self.freeze_these:
                    c_f.set_requires_grad(v, requires_grad=False)
                    v.eval()
                else:
                    v.train()
        self.maybe_freeze_trunk_batchnorm()

    def set_to_eval(self):
        for T in self.trainable_attributes():
            for v in T.values():
                v.eval()

    def initialize_loss_weights(self):
        if self.loss_weights is None:
            self.loss_weights = {k: 1 for k in self.loss_names}

    def initialize_hooks(self):
        if self.end_of_iteration_hook is None:
            self.end_of_iteration_hook = c_f.return_input
        if self.end_of_epoch_hook is None:
            self.end_of_epoch_hook = c_f.return_input

    def initialize_lr_schedulers(self):
        if self.lr_schedulers is None:
            self.lr_schedulers = {}

    def initialize_models(self):
        if "embedder" not in self.models:
            self.models["embedder"] = c_f.Identity()

    def verify_dict_keys(self):
        self.allowed_lr_scheduler_key_suffixes = {
            "iteration": "_scheduler_by_iteration",
            "epoch": "_scheduler_by_epoch",
            "plateau": "_scheduler_by_plateau",
        }
        self.verify_models_keys()
        self.verify_optimizers_keys()
        self.verify_loss_funcs_keys()
        self.verify_mining_funcs_keys()
        self.verify_lr_schedulers_keys()
        self.verify_loss_weights_keys()
        self.verify_gradient_clippers_keys()
        self.verify_freeze_these_keys()

    def _verify_dict_keys(
        self,
        obj_name,
        allowed_keys,
        warn_if_empty,
        important_keys=(),
        essential_keys=(),
    ):
        obj = getattr(self, obj_name, None)
        if obj in [None, {}]:
            if warn_if_empty:
                logging.warning("%s is empty" % obj_name)
        else:
            for k in obj.keys():
                assert any(
                    pattern.match(k) for pattern in c_f.regex_wrapper(allowed_keys)
                ), "%s keys must be one of %s" % (obj_name, ", ".join(allowed_keys))
            for imp_key in important_keys:
                if not any(c_f.regex_wrapper(imp_key).match(k) for k in obj):
                    logging.warning('%s is missing "%s"' % (obj_name, imp_key))
            for ess_key in essential_keys:
                assert any(
                    c_f.regex_wrapper(ess_key).match(k) for k in obj
                ), '%s must contain "%s"' % (obj_name, ess_key)

    def allowed_model_keys(self):
        return ["trunk", "embedder"]

    def allowed_optimizer_keys(self):
        return [
            x + "_optimizer"
            for x in self.allowed_model_keys() + self.allowed_loss_funcs_keys()
        ]

    def allowed_loss_funcs_keys(self):
        return ["metric_loss"]

    def allowed_mining_funcs_keys(self):
        return ["subset_batch_miner", "tuple_miner"]

    def allowed_lr_scheduers_keys(self):
        return [
            x + y
            for y in self.allowed_lr_scheduler_key_suffixes.values()
            for x in self.allowed_model_keys() + self.allowed_loss_funcs_keys()
        ]

    def allowed_gradient_clippers_keys(self):
        return [
            x + "_grad_clipper"
            for x in self.allowed_model_keys() + self.allowed_loss_funcs_keys()
        ]

    def allowed_freeze_these_keys(self):
        return self.allowed_model_keys() + self.allowed_loss_funcs_keys()

    def verify_models_keys(self):
        self._verify_dict_keys(
            "models",
            self.allowed_model_keys(),
            warn_if_empty=True,
            essential_keys=["trunk"],
            important_keys=[x for x in self.allowed_model_keys() if x != "trunk"],
        )

    def verify_optimizers_keys(self):
        self._verify_dict_keys(
            "optimizers",
            self.allowed_optimizer_keys(),
            warn_if_empty=True,
            important_keys=[x + "_optimizer" for x in self.models.keys()],
        )

    def verify_loss_funcs_keys(self):
        self._verify_dict_keys(
            "loss_funcs",
            self.allowed_loss_funcs_keys(),
            warn_if_empty=True,
            important_keys=self.allowed_loss_funcs_keys(),
        )

    def verify_mining_funcs_keys(self):
        self._verify_dict_keys(
            "mining_funcs", self.allowed_mining_funcs_keys(), warn_if_empty=False
        )

    def verify_lr_schedulers_keys(self):
        self._verify_dict_keys(
            "lr_schedulers", self.allowed_lr_scheduers_keys(), warn_if_empty=False
        )

    def verify_loss_weights_keys(self):
        self._verify_dict_keys(
            "loss_weights",
            self.loss_names,
            warn_if_empty=False,
            essential_keys=self.loss_names,
        )

    def verify_gradient_clippers_keys(self):
        self._verify_dict_keys(
            "gradient_clippers",
            self.allowed_gradient_clippers_keys(),
            warn_if_empty=False,
        )

    def verify_freeze_these_keys(self):
        for k in self.freeze_these:
            assert (
                k in self.allowed_freeze_these_keys()
            ), "freeze_these keys must be one of {}".format(
                ", ".join(self.allowed_freeze_these_keys())
            )
            if k + "_optimizer" in self.optimizers.keys():
                logging.warning(
                    "You have passed in an optimizer for {}, but are freezing its parameters.".format(
                        k
                    )
                )
