import numpy as np
from collections import Counter

class TreeNode:
    def __init__(self):
        self.samples = None
        self.category = 0
        self.target_attribute = 0
        self.attribute_list = []
        self.metrics = "Geni index"
        self.is_leaf = False
        self.child_node_list = []

    def set_samples(self,_samples):
        self.samples = _samples

    def set_attribute_list(self,_attributes_list):
        self.attribute_list = _attributes_list

    def set_category(self,_category):
        self.category = _category

    def set_target_attribute(self,_target_attribute):
        self.target_attribute = _target_attribute

    def set_is_leaf(self,_is_leaf):
        self.is_leaf = _is_leaf

    def is_belong_to_same_category(self):
        '''
        samples must not be none
        return (belong_to_same_category, same_label)
        '''
        label_set = set(self.samples[:,-1].tolist())
        if len(label_set) == 1:
            return True , list(label_set)[0]
        else:
            return False , None


    def is_have_same_value_on_attribute_list(self):
        '''
        samples and attribute_list must not be none
        '''
        flag = True
        if len(self.attribute_list) == 0:
            return True
        for attribute_index in self.attribute_list:
            if len(set(self.samples[:,attribute_index].tolist())) != 1:
                flag = False
                break
        return flag

    def get_label_of_most_frequent_samples(self):
        '''
        TODO: what if more than one most common label?
        '''
        most_common_label_list = Counter(self.samples[:,-1].tolist()).most_common(1)
        most_common_label,most_common_label_num_sample = most_common_label_list[0]
        return most_common_label

    def get_ent(self,_t_samples):
        '''
        calculate Ent
        '''
        helper_cnt = Counter(_t_samples[:,-1].tolist())
        total_num_samples = _t_samples.shape[0]
        ent_sum = 0
        for cla in helper_cnt:
            ent_sum += (float) (helper_cnt[cla]) / total_num_samples * np.log2((float) (helper_cnt[cla]) / total_num_samples)
        return (-ent_sum)

    def split_by_attribute_internal(self,_potential_attribute):
        attribute_values_samples_mapping_dict = {}
        for sample_index in range(self.samples.shape[0]):
            if self.samples[sample_index][_potential_attribute] in attribute_values_samples_mapping_dict:
                attribute_values_samples_mapping_dict[self.samples[sample_index][_potential_attribute]] = \
                np.concatenate((attribute_values_samples_mapping_dict[self.samples[sample_index][_potential_attribute]],\
                self.samples[sample_index,:].reshape(1,-1)), axis = 0)
            else:
                attribute_values_samples_mapping_dict[self.samples[sample_index][_potential_attribute]] = \
                self.samples[sample_index,:].reshape(1,-1)
        return attribute_values_samples_mapping_dict
        

    def select_best_attribute_to_split(self):
        '''
        tricky
        return index of attribute
        '''
        information_gain_dict = {}
        for potential_attribute in self.attribute_list:
            subsamples_list = self.split_by_attribute_internal(potential_attribute)
            sum_ent = 0
            for subsamples_key in subsamples_list:
                subsamples = subsamples_list[subsamples_key]
                num_subsamples = subsamples.shape[0]
                num_samples = self.samples.shape[0]
                sum_ent += (float) (num_subsamples) / num_samples * self.get_ent(subsamples)
            information_gain_on_attribute = self.get_ent(self.samples) - sum_ent
            information_gain_dict[potential_attribute] = information_gain_on_attribute
        return max(information_gain_dict, key=information_gain_dict.get)

    def split_by_attribute(self, _target_attribute_index, _target_attribute_value):
        '''
        return sub samples
        '''
        

    def add_child_node(self,_tree_node):
        self.child_node_list.append(_tree_node)
    