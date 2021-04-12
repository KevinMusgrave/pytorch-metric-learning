import math
import random
import itertools

import numpy as np
from torch.utils.data.sampler import BatchSampler


# modified from
# https://github.com/kunhe/Deep-Metric-Learning-Baselines/blob/master/datasets.py
class HierarchicalSampler(BatchSampler):
    def __init__(
        self,
        labels,
        batch_size,
        samples_per_class,
        batches_per_super_pair,
        inner_label=0,
        outer_label=1,
    ):
        """
        labels: 2D array, where rows correspond to elements, and columns correspond to the hierarchical labels
        batch_size: because this is a BatchSampler the batch size must be specified
        samples_per_class: number of instances to sample for a specific class. set to 0 if all element in a class
        batches_per_super_pairs: number of batches to create for a pair of categories (or super labels)
        inner_label: columns index corresponding to classes
        outer_label: columns index corresponding to the level of hierarchy for the pairs
        """
        nb_categories = 2

        self.batch_size = batch_size
        self.batches_per_super_pair = batches_per_super_pair
        self.samples_per_class = samples_per_class

        # checks
        assert self.batch_size % nb_categories == 0, "batch_size should be an even number"
        self.half_bs = self.batch_size // nb_categories
        if self.samples_per_class > 0:
            assert self.half_bs % self.samples_per_class == 0, "batch_size not a multiple of samples_per_class"

        all_super_labels = set(labels[:, outer_label])
        self.super_image_lists = {slb: [] for slb in all_super_labels}
        for slb in all_super_labels:
            conditions = labels[:, outer_label] == slb
            all_labels = set(labels[conditions, inner_label])
            for lb in all_labels:
                idxs = list(np.where(conditions & (labels[:, inner_label] == lb))[0])
                cur_cid_list = list(itertools.product([lb], idxs))
                self.super_image_lists[slb].append(cur_cid_list)

        self.super_pairs = list(itertools.combinations(all_super_labels, nb_categories))

    def __iter__(self,):
        self.reshuffle()
        for batch in self.batches:
            yield batch

    def __len__(self,):
        return len(self.batches)

    def reshuffle(self):
        super_images, num_images, cur_pos = {}, {}, {}

        for sid in self.super_image_lists.keys():
            all_imgs_in_super = self.super_image_lists[sid]

            if self.samples_per_class > 0:
                chunks_list = []
                for cls_imgs in all_imgs_in_super:
                    random.shuffle(cls_imgs)
                    num = len(cls_imgs)
                    # take chunks of size `samples_per_class` and append to chunks_list
                    for c in range(math.ceil(num / self.samples_per_class)):
                        inds = [i % num for i in range(c*self.samples_per_class, (c+1)*self.samples_per_class)]
                        chunks_list.append([cls_imgs[i] for i in inds])
                # concat a "list of lists" into a long list
                random.shuffle(chunks_list)
                super_images[sid] = list(itertools.chain.from_iterable(chunks_list))
            else:
                for cls_imgs in all_imgs_in_super:
                    random.shuffle(cls_imgs)  # shuffle images in each class
                # concat a "list of lists" into a long list
                random.shuffle(all_imgs_in_super)
                super_images[sid] = list(itertools.chain.from_iterable(all_imgs_in_super))

            num_images[sid] = len(super_images[sid])
            cur_pos[sid] = 0

        self.batches = []
        # for each pair of super-labels, e.g. (bicycle, chair)
        for pair in self.super_pairs:
            s0, s1 = pair
            # sample `batches_per_super_pair` batches
            for b in range(self.batches_per_super_pair):
                # get half of the batch from each super-label
                ind0 = [(cur_pos[s0]+i) % num_images[s0] for i in range(self.half_bs)]
                ind1 = [(cur_pos[s1]+i) % num_images[s1] for i in range(self.half_bs)]
                cur_batch = [super_images[s0][i] for i in ind0] + [super_images[s1][i] for i in ind1]

                # move pointers and append to list
                cur_pos[s0] = (ind0[-1] + 1) % num_images[s0]
                cur_pos[s1] = (ind1[-1] + 1) % num_images[s1]

                # Added
                cur_batch = [x[1] for x in cur_batch]
                random.shuffle(cur_batch)

                self.batches.append(cur_batch)

        random.shuffle(self.batches)
