#! /usr/bin/env python3

import tqdm
import torch
import numpy as np
from ..utils import stat_utils
from ..utils import common_functions as c_f
from ..utils import AccuracyCalculator
import logging
from sklearn.preprocessing import normalize, StandardScaler
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
        data_and_label_getter=None, 
        label_hierarchy_level=0, 
        end_of_testing_hook=None,
        dataset_labels=None,
        set_min_label_to_zero=False,
        accuracy_calculator=None,
        visualizer=None,
        visualizer_hook=None
    ):
        self.reference_set = reference_set
        self.normalize_embeddings = normalize_embeddings
        self.pca = int(pca) if pca else None
        self.use_trunk_output = use_trunk_output
        self.batch_size = int(batch_size)
        self.data_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if data_device is None else data_device
        self.dataloader_num_workers = dataloader_num_workers
        self.data_and_label_getter = c_f.return_input if data_and_label_getter is None else data_and_label_getter
        self.label_hierarchy_level = label_hierarchy_level
        self.end_of_testing_hook = end_of_testing_hook
        self.dataset_labels = dataset_labels
        self.set_min_label_to_zero = set_min_label_to_zero
        self.accuracy_calculator = accuracy_calculator
        self.visualizer = visualizer
        self.original_visualizer_hook = visualizer_hook
        self.initialize_label_mapper()
        self.initialize_accuracy_calculator()         

    def initialize_label_mapper(self):
        self.label_mapper = c_f.LabelMapper(self.set_min_label_to_zero, self.dataset_labels).map

    def initialize_accuracy_calculator(self):
        if self.accuracy_calculator is None:
            self.accuracy_calculator = AccuracyCalculator()

    def visualizer_hook(self, *args, **kwargs):
        if self.original_visualizer_hook is not None:
            self.original_visualizer_hook(*args, **kwargs)

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
                label = c_f.process_label(label, "all", self.label_mapper)
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

    def get_all_embeddings(self, dataset, trunk_model, embedder_model=None, collate_fn=None, eval=True):
        if embedder_model is None: embedder_model = c_f.Identity()
        if eval:
            trunk_model.eval()
            embedder_model.eval()
        dataloader = c_f.get_eval_dataloader(dataset, self.batch_size, self.dataloader_num_workers, collate_fn)
        embeddings, labels = self.compute_all_embeddings(dataloader, trunk_model, embedder_model)
        embeddings = self.maybe_normalize(embeddings)
        return embeddings, labels

    def get_embeddings_for_eval(self, trunk_model, embedder_model, input_imgs):
        trunk_output = trunk_model(input_imgs.to(self.data_device))
        if self.use_trunk_output:
            return trunk_output
        return embedder_model(trunk_output)

    def maybe_visualize(self, embeddings_and_labels, epoch):
        self.dim_reduced_embeddings = defaultdict(dict)
        if self.visualizer:
            visualizer_name = self.visualizer.__class__.__name__
            for split_name, (embeddings, labels) in embeddings_and_labels.items():
                logging.info("Running {} on the {} set".format(visualizer_name, split_name))
                dim_reduced = self.visualizer.fit_transform(embeddings)
                logging.info("Finished {}".format(visualizer_name))
                for L in self.label_levels_to_evaluate(labels):
                    label_scheme = labels[:, L]
                    keyname = self.accuracies_keyname(visualizer_name, label_hierarchy_level=L)
                    self.dim_reduced_embeddings[split_name][keyname] = (dim_reduced, label_scheme)
                    self.visualizer_hook(self.visualizer, dim_reduced, label_scheme, split_name, keyname, epoch)

    def description_suffixes(self, base_name):
        if self.pca:
            base_name += "_pca%d"%self.pca
        if self.normalize_embeddings:
            base_name += "_normalized"
        if self.use_trunk_output:
            base_name += "_trunk"
        base_name += "_"+self.reference_set
        base_name += "_"+self.__class__.__name__
        base_name += "_level_"+self.label_hierarchy_level_to_str(self.label_hierarchy_level)
        accuracy_calculator_descriptor = self.accuracy_calculator.description()
        if accuracy_calculator_descriptor != "":
            base_name += "_"+accuracy_calculator_descriptor
        return base_name

    def label_hierarchy_level_to_str(self, label_hierarchy_level):
        if c_f.is_list_or_tuple(label_hierarchy_level):
            return "_".join(str(x) for x in label_hierarchy_level)
        else:
            return str(label_hierarchy_level)

    def accuracies_keyname(self, metric, label_hierarchy_level=0, average=False):
        if average:
            return "AVERAGE_%s"%metric
        if (label_hierarchy_level=="all" or c_f.is_list_or_tuple(label_hierarchy_level)) and len(self.label_levels) == 1:
            label_hierarchy_level = self.label_levels[0]
        return "%s_level%s"%(metric, self.label_hierarchy_level_to_str(label_hierarchy_level))

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
        num_levels_available = query_labels.shape[1]
        if self.label_hierarchy_level == "all":
            return range(num_levels_available)
        elif isinstance(self.label_hierarchy_level, int):
            assert self.label_hierarchy_level < num_levels_available
            return [self.label_hierarchy_level]
        elif c_f.is_list_or_tuple(self.label_hierarchy_level):
            assert max(self.label_hierarchy_level) < num_levels_available
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
        self.embeddings_and_labels = self.get_all_embeddings_for_all_splits(dataset_dict, trunk_model, embedder_model, splits_to_compute_embeddings, collate_fn)
        self.maybe_visualize(self.embeddings_and_labels, epoch)
        self.all_accuracies = defaultdict(dict)
        for split_name in splits_to_eval:
            logging.info('Computing accuracy for the %s split'%split_name)
            self.all_accuracies[split_name]["epoch"] = epoch 
            self.do_knn_and_accuracies(self.all_accuracies[split_name], self.embeddings_and_labels, split_name)
        self.end_of_testing_hook(self) if self.end_of_testing_hook else logging.info(self.all_accuracies)