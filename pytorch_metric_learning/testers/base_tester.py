#! /usr/bin/env python3

import tqdm

import torch
import numpy as np
from ..utils import stat_utils
from ..utils import common_functions as c_f



class BaseTester:
    def __init__(self, k=8, reference_set="query_set", normalize_embeddings=True, use_trunk_output=False, 
                    batch_size=32, dataloader_num_workers=32, split_for_best_epoch="val", 
                    metric_for_best_epoch="recall_at_1", data_device=None, record_keeper=None):
        self.k = int(k)
        self.reference_set = reference_set
        self.normalize_embeddings = normalize_embeddings
        self.use_trunk_output = use_trunk_output
        self.batch_size = int(batch_size)
        self.split_for_best_epoch = split_for_best_epoch
        self.metric_for_best_epoch = metric_for_best_epoch
        self.data_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if data_device is None else data_device
        self.num_workers = dataloader_num_workers
        self.record_keeper = record_keeper

    def calculate_best_epoch(self, all_accuracies):
        if len(all_accuracies) == 0:
            self.accuracies["best_epoch"] = self.accuracies["epoch"]
        else:
            for k, v in all_accuracies.items():
                if self.split_for_best_epoch in k and self.metric_for_best_epoch in k:
                    if self.accuracies[k] > np.max(v):
                        self.accuracies["best_epoch"] = self.accuracies["epoch"]
                    else:
                        self.accuracies["best_epoch"] = all_accuracies["epoch"][np.argmax(v)]
                    break

    def maybe_normalize(self, embeddings):
        if self.normalize_embeddings:
            return embeddings / np.expand_dims(np.linalg.norm(embeddings, axis=1), 1)
        return embeddings

    def compute_all_embeddings(self, dataloader, trunk_model, embedder_model, post_processor):
        num_batches = len(dataloader)
        s, e = 0, 0
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(dataloader)):
                input_imgs = c_f.try_keys(data, ["data", "image"])
                label = data["label"]
                q = self.get_embeddings_for_eval(trunk_model, embedder_model, input_imgs)
                label = c_f.numpy_to_torch(label)
                q, label = post_processor(q, label)
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

    def get_all_embeddings(self, dataset, trunk_model, embedder_model, post_processor, collate_fn):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )
        if post_processor is None: post_processor = lambda e, l: (e, l)
        embeddings, labels = self.compute_all_embeddings(dataloader, trunk_model, embedder_model, post_processor)
        embeddings = self.maybe_normalize(embeddings)
        return embeddings, labels

    def get_embeddings_for_eval(self, trunk_model, embedder_model, input_imgs):
        trunk_output = c_f.pass_data_to_model(trunk_model, input_imgs, self.data_device)
        if self.use_trunk_output:
            return trunk_output
        return embedder_model(trunk_output)

    def suffixes(self, base_name):
        if self.normalize_embeddings:
            base_name += "_normalized"
        if self.use_trunk_output:
            base_name += "_trunk"
        base_name += "_"+self.reference_set
        return base_name

    def accuracies_keyname(self, measure_name, curr_split, label_level):
        measure_name += "_%s" % curr_split
        measure_name += "_level%d" % label_level
        return self.suffixes(measure_name)

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

    def do_knn_and_accuracies(self, embeddings_and_labels, accuracies, epoch, split_keys):
        raise NotImplementedError

    def test(self, dataset_dict, epoch, trunk_model, embedder_model, post_processor=None, collate_fn=None):
        print("Evaluating epoch %d" % epoch)
        trunk_model = trunk_model.eval()
        embedder_model = embedder_model.eval()
        split_keys = list(dataset_dict.keys())
        embeddings_and_labels = {k: None for k in split_keys}
        for split_name, dataset in dataset_dict.items():
            print('Getting embeddings')
            embeddings_and_labels[split_name] = self.get_all_embeddings(dataset, trunk_model, embedder_model, post_processor, collate_fn)
        self.accuracies = {"epoch":epoch}
        self.do_knn_and_accuracies(embeddings_and_labels, epoch, split_keys)
        if self.record_keeper is not None:
            record_group_name = self.suffixes("%s_%s"%("accuracies", self.__class__.__name__))
            self.calculate_best_epoch(self.record_keeper.get_record(record_group_name))
            self.record_keeper.update_records(self.accuracies, epoch, input_group_name_for_non_objects=record_group_name)
        else:
            print(self.accuracies)