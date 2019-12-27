#! /usr/bin/env python3

import tqdm

import torch
import numpy as np
from ..utils import stat_utils
from ..utils import common_functions as c_f
import logging
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.manifold import TSNE


class BaseTester:
    def __init__(self, reference_set="compared_to_self", normalize_embeddings=True, use_trunk_output=False, 
                    batch_size=32, dataloader_num_workers=32, metric_for_best_epoch="mean_average_r_precision", 
                    pca=None, data_device=None, record_keeper=None, size_of_tsne=0, data_and_label_getter=None,
                    label_hierarchy_level=0):
        self.reference_set = reference_set
        self.normalize_embeddings = normalize_embeddings
        self.pca = int(pca) if pca else None
        self.use_trunk_output = use_trunk_output
        self.batch_size = int(batch_size)
        self.metric_for_best_epoch = metric_for_best_epoch
        self.data_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if data_device is None else data_device
        self.num_workers = dataloader_num_workers
        self.record_keeper = record_keeper
        self.size_of_tsne = size_of_tsne
        self.data_and_label_getter = (lambda x : x) if data_and_label_getter is None else data_and_label_getter
        self.base_record_group_name = self.suffixes("%s_%s"%("accuracies", self.__class__.__name__))
        self.label_hierarchy_level = label_hierarchy_level

    def get_accuracy_of_epoch(self, split_name, epoch):
        try:
            records = self.record_keeper.get_record(self.record_group_name(split_name))
            average_key = self.accuracies_keyname(self.metric_for_best_epoch, prefix="AVERAGE")
            if average_key in records:
                return records[average_key][records["epoch"].index(epoch)]
            else:
                for metric, accuracies in records.items():
                    if metric.startswith(self.metric_for_best_epoch):
                        return accuracies[records["epoch"].index(epoch)]
        except:
            return None 

    def get_best_epoch_and_accuracy(self, split_name):
        records = self.record_keeper.get_record(self.record_group_name(split_name))
        best_epoch, best_accuracy = 0, 0
        for epoch in records["epoch"]:
            accuracy = self.get_accuracy_of_epoch(split_name, epoch)
            if accuracy is None:
                return None, None
            if accuracy > best_accuracy:
                best_epoch, best_accuracy = epoch, accuracy
        return best_epoch, best_accuracy

    def maybe_normalize(self, embeddings):
        if self.pca:
            for_pca = StandardScaler().fit_transform(embeddings)
            embeddings = stat_utils.run_pca(for_pca, self.pca)
        if self.normalize_embeddings:
            embeddings = normalize(embeddings)
        return embeddings

    def compute_all_embeddings(self, dataloader, trunk_model, embedder_model):
        num_batches = len(dataloader)
        s, e = 0, 0
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(dataloader)):
                img, label = self.data_and_label_getter(data)
                q = self.get_embeddings_for_eval(trunk_model, embedder_model, img)
                label = c_f.numpy_to_torch(label)
                if label.dim() == 1:
                    label = label.unsqueeze(1)
                if i == 0:
                    labels = torch.zeros(len(dataloader.dataset), label.size(1))
                    all_q = torch.zeros(len(dataloader.dataset), q.size(1))
                e = s + q.size(0)
                all_q[s:e] = q
                labels[s:e] = label
                s = e
            labels = labels.cpu().numpy()
            all_q = all_q.cpu().numpy()

        return all_q, labels

    def get_all_embeddings(self, dataset, trunk_model, embedder_model, collate_fn):
        dataloader = c_f.get_eval_dataloader(dataset, self.batch_size, self.num_workers, collate_fn)
        embeddings, labels = self.compute_all_embeddings(dataloader, trunk_model, embedder_model)
        embeddings = self.maybe_normalize(embeddings)
        return embeddings, labels

    def get_embeddings_for_eval(self, trunk_model, embedder_model, input_imgs):
        trunk_output = c_f.pass_data_to_model(trunk_model, input_imgs, self.data_device)
        if self.use_trunk_output:
            return trunk_output
        return embedder_model(trunk_output)

    def maybe_plot_tsne(self, embeddings_and_labels, epoch, tag_suffix=''):
        if self.size_of_tsne > 0 and self.record_keeper is not None:
            for split_name, (embeddings, labels) in embeddings_and_labels.items():
                random_idx = c_f.NUMPY_RANDOM_STATE.choice(np.arange(len(embeddings)), size=self.size_of_tsne, replace=False)
                curr_embeddings, curr_labels = embeddings[random_idx], labels[random_idx]
                logging.info("Running TSNE on the %s set"%split_name)
                curr_embeddings = TSNE().fit_transform(curr_embeddings)
                logging.info("Finished TSNE")
                for bbb in self.label_levels_to_evaluate(curr_labels):
                    label_scheme = curr_labels[:, bbb]
                    tag = '%s/%s'%(self.record_group_name(split_name), self.accuracies_keyname("tsne", suffix="level%d"%bbb if tag_suffix == '' else tag_suffix))
                    self.record_keeper.add_embedding_plot(curr_embeddings, label_scheme, tag, epoch)


    def suffixes(self, base_name):
        if self.pca:
            base_name += "_pca%d"%self.pca
        if self.normalize_embeddings:
            base_name += "_normalized"
        if self.use_trunk_output:
            base_name += "_trunk"
        base_name += "_"+self.reference_set
        return base_name

    def accuracies_keyname(self, metric, prefix='', suffix=''):
        if prefix != '':
            metric = "%s_%s"%(prefix, metric)
        if suffix != '':
            metric = "%s_%s"%(metric, suffix)
        return metric

    def all_splits_combined(self, embeddings_and_labels):
        eee, lll = list(zip(*list(embeddings_and_labels.values())))
        curr_embeddings = np.concatenate(eee, axis=0)
        curr_labels = np.concatenate(lll, axis=0)
        return curr_embeddings, curr_labels

    def set_reference_and_query(self, embeddings_and_labels, curr_split):
        query_embeddings, query_labels = embeddings_and_labels[curr_split]
        if self.reference_set == "compared_to_self":
            reference_embeddings, reference_labels = query_embeddings, query_labels
        elif self.reference_set == "compared_to_sets_combined":
            reference_embeddings, reference_labels = self.all_splits_combined(embeddings_and_labels)
        elif self.reference_set == "compared_to_training_set":
            reference_embeddings, reference_labels = embeddings_and_labels["train"]
        else:
            raise BaseException 
        return query_embeddings, query_labels, reference_embeddings, reference_labels

    def embeddings_come_from_same_source(self, embeddings_and_labels):
        return self.reference_set in ["compared_to_self", "compared_to_sets_combined"]

    def record_group_name(self, split_name):
        return "%s_%s"%(self.base_record_group_name, split_name.upper())

    def label_levels_to_evaluate(self, query_labels):
        if self.label_hierarchy_level == "all":
            return range(query_labels.shape[1])
        elif query_labels.shape[1] == 1:
            return [0]
        elif isinstance(self.label_hierarchy_level, int):
            return [self.label_hierarchy_level]
        elif isinstance(self.label_hierarchy_level, list):
            return self.label_hierarchy_level

    def calculate_average_accuracies(self, accuracies, metrics):
        for m in metrics:
            keyname = self.accuracies_keyname(m, prefix="AVERAGE")
            summed_accuracy, num_entries = 0, 0
            for metric, value in accuracies.items():
                if metric.startswith(m):
                    summed_accuracy += value
                    num_entries += 1
            accuracies[keyname] = summed_accuracy / num_entries

    def get_splits_to_eval(self, input_splits_to_eval, epoch):
        splits_to_eval = []
        for split in input_splits_to_eval:
            if self.get_accuracy_of_epoch(split, epoch) is None:
                splits_to_eval.append(split)
        return splits_to_eval

    def do_knn_and_accuracies(self, accuracies, embeddings_and_labels, split_name):
        raise NotImplementedError

    def test(self, dataset_dict, epoch, trunk_model, embedder_model, splits_to_eval=None, collate_fn=None, **kwargs):
        logging.info("Evaluating epoch %d" % epoch)
        trunk_model = trunk_model.eval()
        embedder_model = embedder_model.eval()
        split_keys = list(dataset_dict.keys())
        embeddings_and_labels = {k: None for k in split_keys}
        for split_name, dataset in dataset_dict.items():
            logging.info('Getting embeddings for the %s split'%split_name)
            embeddings_and_labels[split_name] = self.get_all_embeddings(dataset, trunk_model, embedder_model, collate_fn)
        self.maybe_plot_tsne(embeddings_and_labels, epoch)
        splits_to_eval = self.get_splits_to_eval(split_keys if splits_to_eval is None else splits_to_eval, epoch)
        for split_name in splits_to_eval:
            logging.info('Computing accuracy for the %s split'%split_name)
            accuracies = {"epoch":epoch}
            self.do_knn_and_accuracies(accuracies, embeddings_and_labels, split_name)
            if self.record_keeper is not None:
                self.record_keeper.update_records(accuracies, epoch, input_group_name_for_non_objects=self.record_group_name(split_name))
                best_epoch, best_accuracy = self.get_best_epoch_and_accuracy(split_name)
                accuracies = {"best_epoch":best_epoch, "best_accuracy":best_accuracy}
                self.record_keeper.update_records(accuracies, epoch, input_group_name_for_non_objects=self.record_group_name(split_name))
            else:
                logging.info(accuracies)