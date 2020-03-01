import logging
from . import common_functions as c_f
import os
import torch
from . import calculate_accuracies as c_a
from collections import defaultdict

# You can write your own hooks for logging.
# But if you'd like something that just works, then use this HookContainer.
# You'll need to install record-keeper and tensorboard.
# pip install record-keeper tensorboard

class HookContainer: 

    def __init__(self, record_keeper, record_group_name_prefix=None, metric_for_best_epoch="mean_average_r_precision"):
        self.record_keeper = record_keeper
        self.record_group_name_prefix = record_group_name_prefix
        self.saveable_trainer_objects = ["models", "optimizers", "lr_schedulers", "loss_funcs", "mining_funcs"]
        self.metric_for_best_epoch = metric_for_best_epoch 

    ############################################
    ############################################
    ##################  HOOKS  #################
    ############################################
    ############################################

    ### Define the end_of_iteration hook. This will be executed at the end of every iteration. ###
    def end_of_iteration_hook(self, trainer):
        record_these = [[trainer.loss_tracker.losses, {"input_group_name_for_non_objects": "loss_histories"}],
                        [trainer.loss_tracker.loss_weights, {"input_group_name_for_non_objects": "loss_weights"}],
                        [trainer.loss_funcs, {"recursive_types": [torch.nn.Module]}],
                        [trainer.mining_funcs, {}],
                        [trainer.models, {}],
                        [trainer.optimizers, {"custom_attr_func": self.optimizer_custom_attr_func}]]
        for record, kwargs in record_these:
            self.record_keeper.update_records(record, trainer.get_global_iteration(), **kwargs)

    # This hook will be passed into the trainer and will be executed at the end of every epoch.
    def end_of_epoch_hook(self, tester, dataset_dict, model_folder, test_interval=1, validation_split_name="val", patience=None):
        if not os.path.exists(model_folder): os.makedirs(model_folder)
        def actual_hook(trainer):
            continue_training = True
            self.record_keeper.maybe_add_custom_figures_to_tensorboard(trainer.get_global_iteration())
            if trainer.epoch % test_interval == 0:
                best_epoch, curr_accuracy = self.save_models_and_eval(trainer, dataset_dict, model_folder, test_interval, tester, validation_split_name)
                trainer.step_lr_plateau_schedulers(curr_accuracy)
                continue_training = self.patience_remaining(trainer.epoch, best_epoch, patience)   
                self.record_keeper.pickler_and_csver.save_records()
            return continue_training
        return actual_hook

    def end_of_testing_hook(self, tester):
        for split_name, accuracies in tester.all_accuracies.items():
            epoch = accuracies["epoch"]
            self.record_keeper.update_records(accuracies, epoch, input_group_name_for_non_objects=self.record_group_name(tester, split_name))
            best_epoch, best_accuracy, _ = self.get_best_epoch_and_accuracies(tester, split_name, ignore_epoch=None)
            best_dict = {"best_epoch":best_epoch, "best_accuracy":best_accuracy}
            self.record_keeper.update_records(best_dict, epoch, input_group_name_for_non_objects=self.record_group_name(tester, split_name))

        for split_name, tsne in tester.tsne_embeddings.items():
            epoch = tsne.pop("epoch", None)
            for k, (tsne_embeddings, tsne_labels) in tsne.items():
                tag = '%s/%s'%(self.record_group_name(tester, split_name), k)
                self.record_keeper.add_embedding_plot(tsne_embeddings, tsne_labels, tag, epoch)



    ############################################
    ############################################
    ######### MODEL LOADING AND SAVING #########
    ############################################
    ############################################

    def load_latest_saved_models_and_records(self, trainer, model_folder, device=None):
        self.record_keeper.pickler_and_csver.load_records()
        if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resume_epoch = c_f.latest_version(model_folder, "trunk_*.pth") or 0
        if resume_epoch > 0:
            for obj_dict in [getattr(trainer, x, {}) for x in self.saveable_trainer_objects]:
                c_f.load_dict_of_models(obj_dict, resume_epoch, model_folder, device)
        return resume_epoch + 1


    def save_models(self, trainer, model_folder, curr_suffix, prev_suffix=None):
        for obj_dict in [getattr(trainer, x, {}) for x in self.saveable_trainer_objects]:
            c_f.save_dict_of_models(obj_dict, curr_suffix, model_folder)
            if prev_suffix is not None:
                c_f.delete_dict_of_models(obj_dict, prev_suffix, model_folder) 

    def save_models_and_eval(self, trainer, dataset_dict, model_folder, test_interval, tester, validation_split_name, **kwargs):
        epoch = trainer.epoch
        self.save_models(trainer, model_folder, epoch, trainer.epoch-test_interval) # save latest model
        tester.test(dataset_dict, epoch, trainer.models["trunk"], trainer.models["embedder"], list(dataset_dict.keys()), trainer.collate_fn, **kwargs)
        best_epoch, curr_accuracy = self.get_best_epoch_and_curr_accuracy(tester, validation_split_name, epoch)
        if epoch == best_epoch:
            logging.info("New best accuracy!")
            self.save_models(trainer, model_folder, "best") # save best model
        return best_epoch, curr_accuracy



    ############################################
    ############################################
    ##### BEST EPOCH AND ACCURACY TRACKING #####
    ############################################
    ############################################

    def get_primary_metric(self, accuracies, tester):
        if accuracies is None:
            return None
        average_key = tester.accuracies_keyname(self.metric_for_best_epoch, prefix="AVERAGE")
        if average_key in accuracies:
            return accuracies[average_key]
        else:
            for k, v in accuracies.items():
                if k.startswith(self.metric_for_best_epoch):
                    return v

    def get_best_epoch_and_curr_accuracy(self, tester, split_name, epoch):
        curr_accuracies = self.get_accuracies_of_epoch(tester, split_name, epoch)
        curr_accuracy = self.get_primary_metric(curr_accuracies, tester)
        best_epoch = self.get_best_epoch_and_accuracies(tester, split_name)[0]
        return best_epoch, curr_accuracy

    def patience_remaining(self, epoch, best_epoch, patience):
        if patience is not None:
            if epoch - best_epoch > patience:
                logging.info("Validation accuracy has plateaued. Exiting.")
                return False
        return True

    def run_tester_separately(self, tester, dataset_dict, epoch, trunk, embedder, splits_to_eval=None, collate_fn=None):
        splits_to_eval = self.get_splits_to_eval(tester, dataset_dict, epoch, splits_to_eval)
        if len(splits_to_eval) == 0:
            logging.info("Already evaluated")
            return False
        tester.test(dataset_dict, epoch, trunk, embedder, splits_to_eval, collate_fn)
        return True

    def get_accuracies_of_epoch(self, tester, split_name, epoch): 
        try:
            records = self.record_keeper.get_record(self.record_group_name(tester, split_name))
            output = {}
            for metric, accuracies in records.items():
                if any(x in metric for x in c_a.METRICS):
                    output[metric] = accuracies[records["epoch"].index(epoch)]
            return output
        except:
            return None 

    def get_best_epoch_and_accuracies(self, tester, split_name, ignore_epoch=-1):
        records = self.record_keeper.get_record(self.record_group_name(tester, split_name))
        best_epoch, best_accuracy, best_accuracies = 0, 0, defaultdict(float)
        for epoch in records["epoch"]:
            if epoch == ignore_epoch:
                continue
            accuracies = self.get_accuracies_of_epoch(tester, split_name, epoch)
            accuracy = self.get_primary_metric(accuracies, tester)
            if accuracy is None:
                return None, None
            if accuracy > best_accuracy:
                best_epoch, best_accuracy, best_accuracies = epoch, accuracy, accuracies
        return best_epoch, best_accuracy, best_accuracies

    def get_splits_to_eval(self, tester, dataset_dict, epoch, input_splits_to_eval):
        input_splits_to_eval = list(dataset_dict.keys()) if input_splits_to_eval is None else input_splits_to_eval
        splits_to_eval = []
        for split in input_splits_to_eval:
            if self.get_accuracies_of_epoch(tester, split, epoch) in [None, {}]:
                splits_to_eval.append(split)
        return splits_to_eval

    def record_group_name(self, tester, split_name):
        base_record_group_name = "%s_"%self.record_group_name_prefix if self.record_group_name_prefix else ''
        base_record_group_name += tester.suffixes("%s_%s"%("accuracies", tester.__class__.__name__))
        return "%s_%s"%(base_record_group_name, split_name.upper())

    def optimizer_custom_attr_func(self, optimizer):
        return {"lr": optimizer.param_groups[0]["lr"]}



class EmptyContainer:
    def end_of_epoch_hook(self, *args):
        return None
    end_of_iteration_hook = None
    end_of_testing_hook = None



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
        return EmptyContainer()