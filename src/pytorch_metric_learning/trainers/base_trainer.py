import torch
import tqdm

from ..utils import common_functions as c_f
from ..utils import loss_tracker as l_t
from ..utils.key_checker import KeyChecker, KeyCheckerDict


class BaseTrainer:
    def __init__(
        self,
        models,
        optimizers,
        batch_size,
        loss_funcs,
        dataset,
        mining_funcs=None,
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
        dataloader_num_workers=2,
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
        self.dataset = dataset
        self.mining_funcs = {} if mining_funcs is None else mining_funcs
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
            c_f.LOGGER.info("TRAINING EPOCH %d" % self.epoch)
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
        c_f.LOGGER.info("Initializing dataloader")
        self.dataloader = c_f.get_train_dataloader(
            self.dataset,
            self.batch_size,
            self.sampler,
            self.dataloader_num_workers,
            self.collate_fn,
        )
        if not self.iterations_per_epoch:
            self.iterations_per_epoch = len(self.dataloader)
        c_f.LOGGER.info("Initializing dataloader iterator")
        self.dataloader_iter = iter(self.dataloader)
        c_f.LOGGER.info("Done creating dataloader iterator")

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
        data = c_f.to_device(data, device=self.data_device, dtype=self.dtype)
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
            self.data_device = c_f.use_cuda_if_available()

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
        self.set_schema()
        self.schema.verify(self)
        self.verify_freeze_these_keys()

    def modify_schema(self):
        pass

    def set_schema(self):
        self.schema = KeyCheckerDict(
            {
                "models": KeyChecker(["trunk", "embedder"], essential=["trunk"]),
                "loss_funcs": KeyChecker(["metric_loss"]),
                "mining_funcs": KeyChecker(
                    ["subset_batch_miner", "tuple_miner"],
                    warn_empty=False,
                    important=[],
                ),
                "loss_weights": KeyChecker(
                    self.loss_names, warn_empty=False, essential=self.loss_names
                ),
                "optimizers": KeyChecker(
                    lambda s, d: c_f.append_map(
                        d["models"].keys + d["loss_funcs"].keys, "_optimizer"
                    ),
                    important=c_f.append_map(self.models.keys(), "_optimizer"),
                ),
                "lr_schedulers": KeyChecker(
                    lambda s, d: [
                        x + y
                        for y in self.allowed_lr_scheduler_key_suffixes.values()
                        for x in d["models"].keys + d["loss_funcs"].keys
                    ],
                    warn_empty=False,
                    important=[],
                ),
                "gradient_clippers": KeyChecker(
                    lambda s, d: c_f.append_map(
                        d["models"].keys + d["loss_funcs"].keys, "_grad_clipper"
                    ),
                    warn_empty=False,
                    important=[],
                ),
            }
        )
        self.modify_schema()

    def verify_freeze_these_keys(self):
        allowed_keys = self.schema["models"].keys + self.schema["loss_funcs"].keys
        for k in self.freeze_these:
            assert k in allowed_keys, "freeze_these keys must be one of {}".format(
                ", ".join(allowed_keys)
            )
            if k + "_optimizer" in self.optimizers.keys():
                c_f.LOGGER.warning(
                    "You have passed in an optimizer for {}, but are freezing its parameters.".format(
                        k
                    )
                )
