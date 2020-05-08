#! /usr/bin/env python3
from collections import defaultdict
import numpy as np
from ..utils import common_functions as c_f
from .base_tester import BaseTester
import logging


class WithSameParentLabelTester(BaseTester):
    def do_knn_and_accuracies(self, accuracies, embeddings_and_labels, split_name, tag_suffix=''):
        query_embeddings, query_labels, reference_embeddings, reference_labels = self.set_reference_and_query(embeddings_and_labels, split_name)
        self.label_levels = [L for L in self.label_levels_to_evaluate(query_labels) if L < query_labels.shape[1] - 1]
        assert len(self.label_levels) > 0, """There are no valid label levels to evaluate. Make sure you've set label_hierarchy_level correctly. 
            If it is an integer, it must be less than the number of hierarchy levels minus 1."""

        for L in self.label_levels:
            curr_query_parent_labels = query_labels[:, L + 1]
            curr_reference_parent_labels = reference_labels[:, L + 1]
            average_accuracies = defaultdict(list)
            for parent_label in np.unique(curr_query_parent_labels):
                logging.info("Label level {} and parent label {}".format(L, parent_label))
                query_match = curr_query_parent_labels == parent_label
                reference_match = curr_reference_parent_labels == parent_label
                curr_query_labels = query_labels[:, L][query_match]
                curr_reference_labels = reference_labels[:, L][reference_match]
                curr_query_embeddings = query_embeddings[query_match]
                curr_reference_embeddings = reference_embeddings[reference_match]
                a = self.accuracy_calculator.get_accuracy(
                    curr_query_embeddings,
                    curr_reference_embeddings,
                    curr_query_labels,
                    curr_reference_labels,
                    self.embeddings_come_from_same_source(embeddings_and_labels),
                )
                for metric, v in a.items():
                    average_accuracies[metric].append(v)
            for metric, v in average_accuracies.items():
                keyname = self.accuracies_keyname(metric, label_hierarchy_level=L)
                accuracies[keyname] = np.mean(v)

        if len(self.label_levels) > 1:
            self.calculate_average_accuracies(accuracies, self.accuracy_calculator.get_curr_metrics(), self.label_levels)
        