import logging

# You can write your own hooks for logging.
# But if you'd like something that just works, then use this HookContainer.
# You'll need to install record-keeper and tensorboard.
# pip install record-keeper

class HookContainer: 

    def __init__(self, record_keeper, record_group_name_prefix=None, end_of_epoch_test=True):
        self.record_keeper = record_keeper
        self.end_of_epoch_test = end_of_epoch_test
        self.record_group_name_prefix = record_group_name_prefix

    # Define the end_of_iteration hook. This will be executed at the end of every iteration.
    # In this example, we'll just log data using record_keeper (if the record-keeper package is installed.)
    def end_of_iteration_hook(self, trainer):
        record_these = [[trainer.loss_tracker.losses, {"input_group_name_for_non_objects": "loss_histories"}],
                        [trainer.loss_tracker.loss_weights, {"input_group_name_for_non_objects": "loss_weights"}],
                        [trainer.loss_funcs, {}],
                        [trainer.mining_funcs, {}],
                        [trainer.models, {}],
                        [trainer.optimizers, {"custom_attr_func": lambda x: {"lr": x.param_groups[0]["lr"]}}]]
        for record, kwargs in record_these:
            self.record_keeper.update_records(record, trainer.get_global_iteration(), **kwargs)

    # This hook will be passed into the trainer and will be executed at the end of every epoch.
    # Note that record_keeper is a custom logging class defined further up in this file.
    def end_of_epoch_hook(self, tester, dataset_dict):
        def actual_hook(trainer):
            if self.end_of_epoch_test:
                tester.test(dataset_dict, trainer.epoch, trainer.models["trunk"], trainer.models["embedder"], list(dataset_dict.keys()), trainer.collate_fn)
            self.record_keeper.maybe_add_custom_figures_to_tensorboard(trainer.get_global_iteration())
            self.record_keeper.pickler_and_csver.save_records()
        return actual_hook

    def end_of_testing_hook(self, tester):
        for split_name, accuracies in tester.all_accuracies.items():
            epoch = accuracies["epoch"]
            self.record_keeper.update_records(accuracies, epoch, input_group_name_for_non_objects=self.record_group_name(tester, split_name))
            best_epoch, best_accuracy = self.get_best_epoch_and_accuracy(tester, split_name, ignore_epoch=None)
            best_dict = {"best_epoch":best_epoch, "best_accuracy":best_accuracy}
            self.record_keeper.update_records(best_dict, epoch, input_group_name_for_non_objects=self.record_group_name(tester, split_name))

        for split_name, tsne in tester.tsne_embeddings.items():
            epoch = tsne.pop("epoch", None)
            for k, (tsne_embeddings, tsne_labels) in tsne.items():
                tag = '%s/%s'%(self.record_group_name(tester, split_name), k)
                self.record_keeper.add_embedding_plot(tsne_embeddings, tsne_labels, tag, epoch)

    def run_tester_separately(self, tester, dataset_dict, epoch, trunk, embedder, splits_to_eval=None, collate_fn=None, **kwargs):
        splits_to_eval = self.get_splits_to_eval(tester, dataset_dict, epoch, splits_to_eval)
        if len(splits_to_eval) == 0:
            logging.info("Already evaluated")
            return False
        tester.test(dataset_dict, epoch, trunk, embedder, splits_to_eval, collate_fn, **kwargs)
        return True

    def get_accuracy_of_epoch(self, tester, split_name, epoch):
        try:
            records = self.record_keeper.get_record(self.record_group_name(tester, split_name))
            average_key = tester.accuracies_keyname(tester.metric_for_best_epoch, prefix="AVERAGE")
            if average_key in records:
                return records[average_key][records["epoch"].index(epoch)]
            else:
                for metric, accuracies in records.items():
                    if metric.startswith(tester.metric_for_best_epoch):
                        return accuracies[records["epoch"].index(epoch)]
        except:
            return None 

    def get_best_epoch_and_accuracy(self, tester, split_name, ignore_epoch=-1):
        records = self.record_keeper.get_record(self.record_group_name(tester, split_name))
        best_epoch, best_accuracy = 0, 0
        for epoch in records["epoch"]:
            if epoch == ignore_epoch:
                continue
            accuracy = self.get_accuracy_of_epoch(tester, split_name, epoch)
            if accuracy is None:
                return None, None
            if accuracy > best_accuracy:
                best_epoch, best_accuracy = epoch, accuracy
        return best_epoch, best_accuracy

    def get_splits_to_eval(self, tester, dataset_dict, epoch, input_splits_to_eval):
        input_splits_to_eval = list(dataset_dict.keys()) if input_splits_to_eval is None else input_splits_to_eval
        splits_to_eval = []
        for split in input_splits_to_eval:
            if self.get_accuracy_of_epoch(tester, split, epoch) is None:
                splits_to_eval.append(split)
        return splits_to_eval

    def record_group_name(self, tester, split_name):
        base_record_group_name = "%s_"%self.record_group_name_prefix if self.record_group_name_prefix else ''
        base_record_group_name += tester.suffixes("%s_%s"%("accuracies", tester.__class__.__name__))
        return "%s_%s"%(base_record_group_name, split_name.upper())



def get_record_keeper(pkl_folder, tensorboard_folder):
    try:
        import record_keeper as record_keeper_package
        from torch.utils.tensorboard import SummaryWriter
        pickler_and_csver = record_keeper_package.PicklerAndCSVer(pkl_folder)
        tensorboard_writer = SummaryWriter(log_dir=tensorboard_folder)
        record_keeper = record_keeper_package.RecordKeeper(tensorboard_writer, pickler_and_csver, ["record_these", "learnable_param_names"])
        return record_keeper, pickler_and_csver, tensorboard_writer

    except ModuleNotFoundError as e:
        logging.warn(e)
        return None, None, None


def get_hook_container(record_keeper):
    if record_keeper:
        return HookContainer(record_keeper)
    else:
        logging.warn("No record_keeper, so no preset hooks are being returned.")
        class EmptyContainer:
            end_of_iteration_hook = None
            end_of_epoch_hook = lambda *args: None
            end_of_testing_hook = None
        return EmptyContainer()