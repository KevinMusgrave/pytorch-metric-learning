#! /usr/bin/env python3
from ..utils import common_functions as c_f
from .base_tester import BaseTester
from .global_embedding_space import GlobalEmbeddingSpaceTester
import torch
import tqdm
import numpy as np

class GlobalTwoStreamEmbeddingSpaceTester(GlobalEmbeddingSpaceTester):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.reference_set == "compared_to_self", "compared_to_self is the only supported reference_set type for {}".format(self.__class__.__name__)
        
    def compute_all_embeddings(self, dataloader, trunk_model, embedder_model):
        num_batches = len(dataloader)
        s, e = 0, 0
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(dataloader)):
                anchors, posnegs, label = self.data_and_label_getter(data)
                label = c_f.process_label(label, self.label_hierarchy_level, self.label_mapper)
                a = self.get_embeddings_for_eval(trunk_model, embedder_model, anchors)
                pns = self.get_embeddings_for_eval(trunk_model, embedder_model, posnegs)
                if label.dim() == 1:
                    label = label.unsqueeze(1)
                if i == 0:
                    labels = torch.zeros(len(dataloader.dataset), label.size(1))
                    all_anchors = torch.zeros(len(dataloader.dataset), pns.size(1))
                    all_posnegs = torch.zeros(len(dataloader.dataset), pns.size(1))
                
                e = s + pns.size(0)
                all_anchors[s:e] = a
                all_posnegs[s:e] = pns
                labels[s:e] = label
                s = e
            labels = labels.cpu().numpy()
            all_posnegs = all_posnegs.cpu().numpy()
            all_anchors = all_anchors.cpu().numpy()

        return all_anchors, all_posnegs, labels
    
    def get_all_embeddings(self, dataset, trunk_model, embedder_model, collate_fn):
        dataloader = c_f.get_eval_dataloader(dataset, self.batch_size, self.dataloader_num_workers, collate_fn)
        anchor_embeddings, posneg_embeddings, labels = self.compute_all_embeddings(dataloader, trunk_model, embedder_model)
        anchor_embeddings, posneg_embeddings = self.maybe_normalize(anchor_embeddings), self.maybe_normalize(posneg_embeddings)
        return np.concatenate([anchor_embeddings, posneg_embeddings], axis=0), np.concatenate([labels, labels], axis=0)
    
    def set_reference_and_query(self, embeddings_and_labels, curr_split):
        embeddings, labels = embeddings_and_labels[curr_split]
        half = int(embeddings.shape[0] / 2)
        anchors_embeddings = embeddings[:half]
        posneg_embeddings = embeddings[half:]
        query_labels = labels[:half]
        return anchors_embeddings, query_labels, posneg_embeddings, query_labels
    
    def embeddings_come_from_same_source(self, embeddings_and_labels):
        return False