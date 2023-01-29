from .base_tester import BaseTester


class GlobalEmbeddingSpaceTester(BaseTester):
    def do_knn_and_accuracies(
        self, accuracies, embeddings_and_labels, query_split_name, reference_split_names
    ):
        (
            query_embeddings,
            query_labels,
            reference_embeddings,
            reference_labels,
        ) = self.set_reference_and_query(
            embeddings_and_labels, query_split_name, reference_split_names
        )
        self.label_levels = self.label_levels_to_evaluate(query_labels)

        for L in self.label_levels:
            curr_query_labels = query_labels[:, L]
            curr_reference_labels = reference_labels[:, L]
            a = self.accuracy_calculator.get_accuracy(
                query_embeddings,
                curr_query_labels,
                reference_embeddings,
                curr_reference_labels,
                self.ref_includes_query(query_split_name, reference_split_names),
            )
            for metric, v in a.items():
                keyname = self.accuracies_keyname(metric, label_hierarchy_level=L)
                accuracies[keyname] = v
        if len(self.label_levels) > 1:
            self.calculate_average_accuracies(
                accuracies,
                self.accuracy_calculator.get_curr_metrics(),
                self.label_levels,
            )
