#! /usr/bin/env python3
from ..utils import calculate_accuracies
from ..utils import common_functions as c_f
from .base_tester import BaseTester
from .global_embedding_space import GlobalEmbeddingSpaceTester
import torch
import tqdm

class GlobalTwoStreamEmbeddingSpaceTester(GlobalEmbeddingSpaceTester):

    def compute_all_embeddings(self, dataloader, trunk_model, embedder_model):
        num_batches = len(dataloader)
        s, e = 0, 0
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(dataloader)):
                queries, anchors, label = self.data_and_label_getter(data)
                label = c_f.process_label(label, self.label_hierarchy_level, self.label_mapper)
                a = self.get_embeddings_for_eval(trunk_model, embedder_model, anchors)
                q = self.get_embeddings_for_eval(trunk_model, embedder_model, queries)
                if label.dim() == 1:
                    label = label.unsqueeze(1)
                if i == 0:
                    labels = torch.zeros(len(dataloader.dataset), label.size(1))
                    all_queries = torch.zeros(len(dataloader.dataset), q.size(1))
                    all_anchors = torch.zeros(len(dataloader.dataset), q.size(1))
                
                e = s + q.size(0)
                all_queries[s:e] = q
                all_anchors[s:e] = a
                labels[s:e] = label
                s = e
            labels = labels.cpu().numpy()
            all_queries = all_queries.cpu().numpy()
            all_anchors = all_anchors.cpu().numpy()

        return all_queries, all_anchors, labels
    
    def get_all_embeddings(self, dataset, trunk_model, embedder_model, collate_fn):
        dataloader = c_f.get_eval_dataloader(dataset, self.batch_size, self.dataloader_num_workers, collate_fn)
        query_embeddings,anchor_embeddings, labels = self.compute_all_embeddings(dataloader, trunk_model, embedder_model)
        return self.maybe_normalize(query_embeddings), self.maybe_normalize(anchor_embeddings), labels
    
    def set_reference_and_query(self, embeddings_and_labels, curr_split):
        query_embeddings, anchors_embeddings, query_labels = embeddings_and_labels[curr_split]
        return query_embeddings, query_labels, anchors_embeddings, query_labels
    
    def embeddings_come_from_same_source(self, embeddings_and_labels):
        return False