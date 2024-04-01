from collections import defaultdict

import torch
import tqdm

from ..utils import common_functions as c_f
from ..utils import inference
from ..utils.accuracy_calculator import AccuracyCalculator


class BaseTester:
    def __init__(
        self,
        normalize_embeddings=True,
        use_trunk_output=False,
        batch_size=32,
        dataloader_num_workers=2,
        pca=None,
        data_device=None,
        dtype=None,
        data_and_label_getter=None,
        label_hierarchy_level=0,
        end_of_testing_hook=None,
        dataset_labels=None,
        set_min_label_to_zero=False,
        accuracy_calculator=None,
        visualizer=None,
        visualizer_hook=None,
    ):
        self.normalize_embeddings = normalize_embeddings
        self.pca = int(pca) if pca else None
        self.use_trunk_output = use_trunk_output
        self.batch_size = int(batch_size)
        self.data_device = (
            c_f.use_cuda_if_available() if data_device is None else data_device
        )
        self.dtype = dtype
        self.dataloader_num_workers = dataloader_num_workers
        self.data_and_label_getter = (
            c_f.return_input if data_and_label_getter is None else data_and_label_getter
        )
        self.label_hierarchy_level = label_hierarchy_level
        self.end_of_testing_hook = end_of_testing_hook
        self.dataset_labels = dataset_labels
        self.set_min_label_to_zero = set_min_label_to_zero
        self.accuracy_calculator = accuracy_calculator
        self.visualizer = visualizer
        self.original_visualizer_hook = visualizer_hook
        self.initialize_label_mapper()
        self.initialize_accuracy_calculator()
        self.reference_split_names = {}

    def initialize_label_mapper(self):
        self.label_mapper = c_f.LabelMapper(
            self.set_min_label_to_zero, self.dataset_labels
        ).map

    def initialize_accuracy_calculator(self):
        if self.accuracy_calculator is None:
            self.accuracy_calculator = AccuracyCalculator()

    def visualizer_hook(self, *args, **kwargs):
        if self.original_visualizer_hook is not None:
            self.original_visualizer_hook(*args, **kwargs)

    def maybe_normalize(self, embeddings):
        if self.pca:
            for_pca = c_f.torch_standard_scaler(embeddings)
            embeddings = inference.run_pca(for_pca, self.pca)
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings

    def compute_all_embeddings(self, dataloader, trunk_model, embedder_model):
        s, e = 0, 0
        with torch.no_grad():
            for i, data in enumerate(tqdm.tqdm(dataloader)):
                img, label = self.data_and_label_getter(data)
                label = c_f.process_label(label, "all", self.label_mapper)
                q = self.get_embeddings_for_eval(trunk_model, embedder_model, img)
                if label.dim() == 1:
                    label = label.unsqueeze(1)
                if i == 0:
                    labels = torch.zeros(
                        len(dataloader.dataset),
                        label.size(1),
                        device=self.data_device,
                        dtype=label.dtype,
                    )
                    all_q = torch.zeros(
                        len(dataloader.dataset),
                        q.size(1),
                        device=self.data_device,
                        dtype=q.dtype,
                    )
                e = s + q.size(0)
                all_q[s:e] = q
                labels[s:e] = label
                s = e
        return all_q, labels

    def get_all_embeddings(
        self,
        dataset,
        trunk_model,
        embedder_model=None,
        collate_fn=None,
        eval=True,
        return_as_numpy=False,
    ):
        if embedder_model is None:
            embedder_model = torch.nn.Identity()
        if eval:
            trunk_model.eval()
            embedder_model.eval()
        dataloader = c_f.get_eval_dataloader(
            dataset, self.batch_size, self.dataloader_num_workers, collate_fn
        )
        embeddings, labels = self.compute_all_embeddings(
            dataloader, trunk_model, embedder_model
        )
        embeddings = self.maybe_normalize(embeddings)
        if return_as_numpy:
            return embeddings.cpu().numpy(), labels.cpu().numpy()
        return embeddings, labels

    def get_embeddings_for_eval(self, trunk_model, embedder_model, input_imgs):
        input_imgs = c_f.to_device(
            input_imgs, device=self.data_device, dtype=self.dtype
        )
        trunk_output = trunk_model(input_imgs)
        if self.use_trunk_output:
            return trunk_output
        return embedder_model(trunk_output)

    def maybe_visualize(self, embeddings_and_labels, epoch):
        if self.visualizer:
            visualizer_name = self.visualizer.__class__.__name__
            for split_name, (embeddings, labels) in embeddings_and_labels.items():
                c_f.LOGGER.info(
                    "Running {} on the {} set".format(visualizer_name, split_name)
                )
                dim_reduced = self.visualizer.fit_transform(embeddings.cpu().numpy())
                c_f.LOGGER.info("Finished {}".format(visualizer_name))
                for L in self.label_levels_to_evaluate(labels):
                    label_scheme = labels[:, L].cpu().numpy()
                    keyname = self.accuracies_keyname(
                        visualizer_name, label_hierarchy_level=L
                    )
                    self.visualizer_hook(
                        self.visualizer,
                        dim_reduced,
                        label_scheme,
                        split_name,
                        keyname,
                        epoch,
                    )

    def description_suffixes(self, base_name):
        if self.pca:
            base_name += "_pca%d" % self.pca
        if self.normalize_embeddings:
            base_name += "_normalized"
        if self.use_trunk_output:
            base_name += "_trunk"
        base_name += "_" + self.__class__.__name__
        base_name += "_level_" + self.label_hierarchy_level_to_str(
            self.label_hierarchy_level
        )
        accuracy_calculator_descriptor = self.accuracy_calculator.description()
        if accuracy_calculator_descriptor != "":
            base_name += "_" + accuracy_calculator_descriptor
        return base_name

    def label_hierarchy_level_to_str(self, label_hierarchy_level):
        if c_f.is_list_or_tuple(label_hierarchy_level):
            return "_".join(str(x) for x in label_hierarchy_level)
        else:
            return str(label_hierarchy_level)

    def accuracies_keyname(self, metric, label_hierarchy_level=0, average=False):
        if average:
            return "AVERAGE_%s" % metric
        if (
            label_hierarchy_level == "all"
            or c_f.is_list_or_tuple(label_hierarchy_level)
        ) and len(self.label_levels) == 1:
            label_hierarchy_level = self.label_levels[0]
        return "%s_level%s" % (
            metric,
            self.label_hierarchy_level_to_str(label_hierarchy_level),
        )

    def maybe_combine_splits(self, embeddings_and_labels, splits):
        to_combine = {split: embeddings_and_labels[split] for split in splits}
        eee, lll = list(zip(*list(to_combine.values())))
        curr_embeddings = torch.cat(eee, dim=0)
        curr_labels = torch.cat(lll, dim=0)
        return curr_embeddings, curr_labels

    def set_reference_and_query(
        self, embeddings_and_labels, query_split_name, reference_split_names
    ):
        query_embeddings, query_labels = embeddings_and_labels[query_split_name]
        reference_embeddings, reference_labels = self.maybe_combine_splits(
            embeddings_and_labels, reference_split_names
        )
        return query_embeddings, query_labels, reference_embeddings, reference_labels

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
        if splits_to_eval is None:
            splits_to_eval = [(k, [k]) for k in dataset_dict]
        query_splits = [t[0] for t in splits_to_eval]
        assert len(query_splits) == len(
            set(query_splits)
        ), "Unsupported: Evaluating a query split more than once"
        splits_to_compute_embeddings = set()
        for i, (query_split, reference_splits) in enumerate(splits_to_eval):
            splits_to_compute_embeddings.update(reference_splits + [query_split])
            if query_split in reference_splits:
                # AccuracyCalculator requires that query be at the beginning of ref
                reference_splits.remove(query_split)
                splits_to_eval[i] = (query_split, [query_split] + reference_splits)
        return splits_to_eval, list(splits_to_compute_embeddings)

    def get_all_embeddings_for_all_splits(
        self,
        dataset_dict,
        trunk_model,
        embedder_model,
        splits_to_compute_embeddings,
        collate_fn,
    ):
        embeddings_and_labels = {}
        for split_name in splits_to_compute_embeddings:
            c_f.LOGGER.info("Getting embeddings for the %s split" % split_name)
            embeddings_and_labels[split_name] = self.get_all_embeddings(
                dataset_dict[split_name], trunk_model, embedder_model, collate_fn
            )
        return embeddings_and_labels

    def do_knn_and_accuracies(
        self, accuracies, embeddings_and_labels, query_split_name, reference_split_names
    ):
        raise NotImplementedError

    def ref_includes_query(self, query_split_name, reference_split_names):
        return query_split_name in reference_split_names

    def test(
        self,
        dataset_dict,
        epoch,
        trunk_model,
        embedder_model=None,
        splits_to_eval=None,
        collate_fn=None,
    ):
        c_f.LOGGER.info("Evaluating epoch {}".format(epoch))
        if embedder_model is None:
            embedder_model = torch.nn.Identity()
        trunk_model.eval()
        embedder_model.eval()
        (
            splits_to_eval,
            splits_to_compute_embeddings,
        ) = self.get_splits_to_compute_embeddings(dataset_dict, splits_to_eval)
        self.embeddings_and_labels = self.get_all_embeddings_for_all_splits(
            dataset_dict,
            trunk_model,
            embedder_model,
            splits_to_compute_embeddings,
            collate_fn,
        )
        self.maybe_visualize(self.embeddings_and_labels, epoch)
        self.all_accuracies = defaultdict(dict)
        for query_split_name, reference_split_names in splits_to_eval:
            c_f.LOGGER.info(
                f"Computing accuracy for the {query_split_name} split w.r.t {reference_split_names}"
            )
            self.all_accuracies[query_split_name]["epoch"] = epoch
            self.reference_split_names[query_split_name] = reference_split_names
            self.do_knn_and_accuracies(
                self.all_accuracies[query_split_name],
                self.embeddings_and_labels,
                query_split_name,
                reference_split_names,
            )
        (
            self.end_of_testing_hook(self)
            if self.end_of_testing_hook
            else c_f.LOGGER.info(self.all_accuracies)
        )
        del self.embeddings_and_labels
        return self.all_accuracies
