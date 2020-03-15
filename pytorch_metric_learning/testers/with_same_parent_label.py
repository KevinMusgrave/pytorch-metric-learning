#! /usr/bin/env python3
from collections import defaultdict

import numpy as np
from ..utils import calculate_accuracies
from ..utils import common_functions as c_f

from .base_tester import BaseTester


class WithSameParentLabelTester(BaseTester):
    def do_knn_and_accuracies(self, accuracies, embeddings_and_labels, split_name, tag_suffix=''):
        query_embeddings, query_labels, reference_embeddings, reference_labels = self.set_reference_and_query(embeddings_and_labels, split_name)
        label_levels = self.label_levels_to_evaluate(query_labels)

        for bbb in label_levels:
            if bbb + 1 >= query_labels.shape[1]:
                continue
            curr_query_parent_labels = query_labels[:, bbb + 1]
            curr_reference_parent_labels = reference_labels[:, bbb + 1]
            average_accuracies = defaultdict(list)
            for parent_label in np.unique(curr_query_parent_labels):
                query_match = curr_query_parent_labels == parent_label
                reference_match = curr_reference_parent_labels == parent_label
                curr_query_labels = query_labels[:, bbb][query_match]
                curr_reference_labels = reference_labels[:, bbb][reference_match]
                curr_query_embeddings = query_embeddings[query_match]
                curr_reference_embeddings = reference_embeddings[reference_match]
                a = calculate_accuracies.calculate_accuracy(
                    curr_query_embeddings,
                    curr_reference_embeddings,
                    curr_query_labels,
                    curr_reference_labels,
                    self.embeddings_come_from_same_source(embeddings_and_labels),
                )
                for metric, v in a.items():
                    average_accuracies[metric].append(v)
            for metric, v in average_accuracies.items():
                keyname = self.accuracies_keyname(metric, label_hierarchy_level=bbb)
                accuracies[keyname].append(np.mean(v))
                
        if len(label_levels) > 1:
            self.calculate_average_accuracies(accuracies, calculate_accuracies.METRICS, label_levels)