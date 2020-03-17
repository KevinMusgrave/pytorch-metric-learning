#! /usr/bin/env python3

import tqdm

import torch
import numpy as np
from ..utils import stat_utils
from ..utils import common_functions as c_f
import logging
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.manifold import TSNE
from collections import defaultdict


class BaseTester:
    def __init__(
        self, 
        reference_set="compared_to_self", 
        normalize_embeddings=True, 
        use_trunk_output=False, 
        batch_size=32, 
        dataloader_num_workers=32, 
        pca=None, 
        data_device=None, 
        size_of_tsne=0, 
        data_and_label_getter=None, 
        label_hierarchy_level=0, 
        end_of_testing_hook=None,
        dataset_labels=None,
        set_min_label_to_zero=False
    ):
        self.reference_set = reference_set
        self.normalize_embeddings = normalize_embeddings
        self.pca = int(pca) if pca else None
        self.use_trunk_output = use_trunk_output
        self.batch_size = int(batch_size)
        self.data_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if data_device is None else data_device
        self.dataloader_num_workers = dataloader_num_workers
        self.size_of_tsne = size_of_tsne
        self.data_and_label_getter = c_f.return_input if data_and_label_getter is None else data_and_label_getter
        self.label_hierarchy_level = label_hierarchy_level
        self.end_of_testing_hook = end_of_testing_hook
        self.dataset_labels = dataset_labels
        self.set_min_label_to_zero = set_min_label_to_zero
        self.initialize_label_mapper()              

    def initialize_label_mapper(self):
        self.label_mapper = c_f.LabelMapper(self.set_min_label_to_zero, self.dataset_labels).map

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
                label = c_f.process_label(label, self.label_hierarchy_level, self.label_mapper)
                q = self.get_embeddings_for_eval(trunk_model, embedder_model, img)
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
        dataloader = c_f.get_eval_dataloader(dataset, self.batch_size, self.dataloader_num_workers, collate_fn)
        embeddings, labels = self.compute_all_embeddings(dataloader, trunk_model, embedder_model)
        embeddings = self.maybe_normalize(embeddings)
        return embeddings, labels

    def get_embeddings_for_eval(self, trunk_model, embedder_model, input_imgs):
        trunk_output = c_f.pass_data_to_model(trunk_model, input_imgs, self.data_device)
        if self.use_trunk_output:
            return trunk_output
        return embedder_model(trunk_output)

    def maybe_compute_tsne(self, embeddings_and_labels, epoch):
        self.tsne_embeddings = defaultdict(dict)
        if self.size_of_tsne > 0:
            for split_name, (embeddings, labels) in embeddings_and_labels.items():
                random_idx = c_f.NUMPY_RANDOM.choice(np.arange(len(embeddings)), size=self.size_of_tsne, replace=False)
                curr_embeddings, curr_labels = embeddings[random_idx], labels[random_idx]
                logging.info("Running TSNE on the %s set"%split_name)
                curr_embeddings = TSNE().fit_transform(curr_embeddings)
                logging.info("Finished TSNE")
                self.tsne_embeddings[split_name]["epoch"] = epoch
                for bbb in self.label_levels_to_evaluate(curr_labels):
                    label_scheme = curr_labels[:, bbb]
                    keyname = self.accuracies_keyname("tsne", label_hierarchy_level=bbb)
                    self.tsne_embeddings[split_name][keyname] = (curr_embeddings, label_scheme)

    def description_suffixes(self, base_name):
        if self.pca:
            base_name += "_pca%d"%self.pca
        if self.normalize_embeddings:
            base_name += "_normalized"
        if self.use_trunk_output:
            base_name += "_trunk"
        base_name += "_"+self.reference_set
        base_name += "_"+self.__class__.__name__
        base_name += "_level_"+str(self.label_hierarchy_level)
        return base_name

    def accuracies_keyname(self, metric, label_hierarchy_level=0, average=False):
        if average:
            return "AVERAGE_%s"%metric
        return "%s_level%d"%(metric, label_hierarchy_level)

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

    def label_levels_to_evaluate(self, query_labels):
        if self.label_hierarchy_level == "all":
            return range(query_labels.shape[1])
        elif query_labels.shape[1] == 1:
            return [0]
        elif isinstance(self.label_hierarchy_level, int):
            return [self.label_hierarchy_level]
        elif isinstance(self.label_hierarchy_level, list):
            return self.label_hierarchy_level

    def calculate_average_accuracies(self, accuracies, metrics, label_levels):
        for m in metrics:
            keyname = self.accuracies_keyname(m, average=True)
            summed_accuracy = 0
            for L in label_levels:
                curr_key = self.accuracies_keyname(m, label_hierarchy_level=L)
                summed_accuracy += accuracies[curr_key]
            accuracies[keyname] = summed_accuracy / len(label_levels)

    def get_splits_to_compute_embeddings(self, dataset_dict, splits_to_eval):
        splits_to_eval = list(dataset_dict.keys()) if splits_to_eval is None else splits_to_eval
        if self.reference_set in ["compared_to_self", "compared_to_sets_combined"]:
            return splits_to_eval, splits_to_eval
        if self.reference_set == "compared_to_training_set":
            return splits_to_eval, list(set(splits_to_eval).add("train"))

    def get_all_embeddings_for_all_splits(self, dataset_dict, trunk_model, embedder_model, splits_to_compute_embeddings, collate_fn):
        embeddings_and_labels = {}
        for split_name in splits_to_compute_embeddings:
            logging.info('Getting embeddings for the %s split'%split_name)
            embeddings_and_labels[split_name] = self.get_all_embeddings(dataset_dict[split_name], trunk_model, embedder_model, collate_fn)
        return embeddings_and_labels

    def do_knn_and_accuracies(self, accuracies, embeddings_and_labels, split_name):
        raise NotImplementedError

    def test(self, dataset_dict, epoch, trunk_model, embedder_model=None, splits_to_eval=None, collate_fn=None, **kwargs):
        logging.info("Evaluating epoch %d" % epoch)
        if embedder_model is None: embedder_model = c_f.Identity()
        trunk_model.eval()
        embedder_model.eval()
        splits_to_eval, splits_to_compute_embeddings = self.get_splits_to_compute_embeddings(dataset_dict, splits_to_eval)
        embeddings_and_labels = self.get_all_embeddings_for_all_splits(dataset_dict, trunk_model, embedder_model, splits_to_compute_embeddings, collate_fn)
        self.maybe_compute_tsne(embeddings_and_labels, epoch)
        self.all_accuracies = defaultdict(dict)
        for split_name in splits_to_eval:
            logging.info('Computing accuracy for the %s split'%split_name)
            self.all_accuracies[split_name]["epoch"] = epoch 
            self.do_knn_and_accuracies(self.all_accuracies[split_name], embeddings_and_labels, split_name)
        self.end_of_testing_hook(self) if self.end_of_testing_hook else logging.info(self.all_accuracies)