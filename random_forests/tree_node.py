import numpy as np

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
        '''
        read np matrix and set samples
        '''
        pass

    def set_attribute_list(self,_attributes_list):
        pass

    def set_category(self,category):
        pass

    def set_target_attribute(self,target_attribute):
        pass

    def is_belong_to_same_category(self):
        '''
        samples must not be none
        return (belong_to_same_category, same_label)
        '''

    def is_have_same_value_on_attribute_list(self):
        '''
        samples must not be none
        '''

    def get_label_of_most_frequent_samples(self):
        pass

    def select_best_attribute_to_split(self):
        '''
        tricky
        return index of attribute
        '''
        pass

    def get_all_possible_values_on_attribute(self,_target_attribute):
        '''
        return a list
        '''

    def split_by_attribute(self,attribute_value):
        '''
        return sub samples
        '''
        pass

    def add_child_node(self,_tree_node):
        self.child_node_list.append(_tree_node)
    