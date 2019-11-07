#! /usr/bin/env python3

from . import common_functions as c_f
import collections
import matplotlib.pyplot as plt
import numpy as np


class RecordKeeper:
    def __init__(self, tensorboard_writer=None, pickler_and_csver=None):
        self.tensorboard_writer = tensorboard_writer
        self.pickler_and_csver = pickler_and_csver

    def append_data(self, group_name, series_name, value, iteration):
        if self.tensorboard_writer is not None:
            tag_name = '%s/%s' % (group_name, series_name)
            if not c_f.is_list_and_has_more_than_one_element(value):
                self.tensorboard_writer.add_scalar(tag_name, value, iteration)
        if self.pickler_and_csver is not None:
            self.pickler_and_csver.append(group_name, series_name, value)

    def update_records(self, record_these, global_iteration, custom_attr_func=None, input_group_name_for_non_objects=None):
        for name_in_dict, input_obj in record_these.items():

            if input_group_name_for_non_objects is not None:
                group_name = input_group_name_for_non_objects
                self.append_data(group_name, name_in_dict, input_obj, global_iteration)
            else:
                the_obj = c_f.try_getting_dataparallel_module(input_obj)
                attr_list = self.get_attr_list_for_record_keeper(the_obj)
                for k in attr_list:
                    v = getattr(the_obj, k)
                    name = self.get_record_name(name_in_dict, the_obj)
                    self.append_data(name, k, v, global_iteration)
                if custom_attr_func is not None:
                    for k, v in custom_attr_func(the_obj).items():
                        name = self.get_record_name(name_in_dict, the_obj)
                        self.append_data(name, k, v, global_iteration)


    def get_attr_list_for_record_keeper(self, input_obj):
        attr_list = []
        obj_attr_list_names = ["record_these", "learnable_param_names"]
        for k in obj_attr_list_names:
            if (hasattr(input_obj, k)) and (getattr(input_obj, k) is not None):
                attr_list += getattr(input_obj, k)
        return attr_list

    def get_record_name(self, name_in_dict, input_obj, key_name=None):
        record_name = "%s_%s" % (name_in_dict, type(input_obj).__name__)
        if key_name:
            record_name += '_%s' % key_name
        return record_name

    def maybe_add_custom_figures_to_tensorboard(self, global_iteration):
        if self.pickler_and_csver is not None:
            for group_name, dict_of_lists in self.pickler_and_csver.records.items():
                for series_name, v in dict_of_lists.items():
                    if isinstance(v[0], list):
                        tag_name = '%s/%s' % (group_name, series_name)
                        figure = self.multi_line_plot(v)
                        self.tensorboard_writer.add_figure(tag_name, figure, global_iteration)

    def multi_line_plot(self, list_of_lists):
        # Each sublist represents a snapshot at an iteration.
        # Transpose so that each row covers many iterations.
        numpified = np.transpose(np.array(list_of_lists))
        fig = plt.figure()
        for sublist in numpified:
            plt.plot(np.arange(numpified.shape[1]), sublist)
        return fig

    def get_record(self, group_name):
        return self.pickler_and_csver.records[group_name]


class PicklerAndCSVer:
    def __init__(self, folder):
        self.records = collections.defaultdict(lambda: collections.defaultdict(list))
        self.folder = folder

    def append(self, group_name, series_name, input_val):
        if c_f.is_list_and_has_more_than_one_element(input_val):
            self.records[group_name][series_name].append(c_f.convert_to_list(input_val))
        else:
            self.records[group_name][series_name].append(c_f.convert_to_scalar(input_val))

    def save_records(self):
        for k, v in self.records.items():
            base_filename = "%s/%s" % (self.folder, k)
            c_f.save_pkl(v, base_filename+".pkl")
            c_f.write_dict_of_lists_to_csv(v, base_filename+".csv")

    def load_records(self, num_records_to_load=None):
        for k, _ in self.records.items():
            filename = "%s/%s.pkl"%(self.folder,k)
            self.records[k] = c_f.load_pkl(filename)
            if num_records_to_load is not None:
                for zzz, _ in self.records[k].items():
                    self.records[k][zzz] = self.records[k][zzz][:num_records_to_load]
