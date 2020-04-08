#! /usr/bin/env python3
from ..utils import calculate_accuracies
from ..utils import common_functions as c_f

from .base_tester import BaseTester


class GlobalEmbeddingSpaceTester(BaseTester):

    def do_knn_and_accuracies(self, accuracies, embeddings_and_labels, split_name):
        query_embeddings, query_labels, reference_embeddings, reference_labels = self.set_reference_and_query(embeddings_and_labels, split_name)
        label_levels = self.label_levels_to_evaluate(query_labels)

        for i, bbb in enumerate(label_levels):
            curr_query_labels = query_labels[:, bbb]
            curr_reference_labels = reference_labels[:, bbb]
            a = calculate_accuracies.calculate_accuracy(
                query_embeddings,
                reference_embeddings,
                curr_query_labels,
                curr_reference_labels,
                self.embeddings_come_from_same_source(embeddings_and_labels),
            )
            for metric, v in a.items():
                keyname = self.accuracies_keyname(metric, label_hierarchy_level=bbb)
                accuracies[keyname] = v

        if len(label_levels) > 1:
            self.calculate_average_accuracies(accuracies, calculate_accuracies.METRICS, label_levels)