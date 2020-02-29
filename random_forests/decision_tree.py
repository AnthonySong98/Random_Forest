import numpy as np
from tree_node import TreeNode

class DecisionTree:
    def __init__(self):
        self.root = TreeNode()
        
    def set_root(self,_tree_node):
        pass

    def generate_decision_tree(_training_samples, _attributes_list):
        tree_node = TreeNode()
        tree_node.set_samples(_training_samples)
        tree_node.set_attribute_list(_attributes_list)

        # test if _training_samples belong to one same category already
        belong_to_same_category, same_label =  tree_node.is_belong_to_same_category()
        if belong_to_same_category == True:
            tree_node.set_is_leaf(True)
            tree_node.category = same_label
            return tree_node
        
        if len(_attributes_list) == 0 or tree_node.is_have_same_value_on_attribute_list():
            tree_node.set_is_leaf(True)
            tree_node.set_category(tree_node.get_label_of_most_frequent_samples())
            return tree_node

        best_attribute_to_split = tree_node.select_best_attribute_to_split()

        # set the target split attribute
        tree_node.set_target_attribute(best_attribute_to_split)

        # TODO: get_all_possible_values_on_attribute
        for possible_value in self.get_all_possible_values_on_attribute(best_attribute_to_split):
            sub_tree_node = TreeNode()
            sub_training_samples = tree_node.split_by_attribute(possible_value)

            if sub_training_samples.shape[0] == 0:
                sub_tree_node.set_samples(sub_training_samples)
                sub_tree_node.set_is_leaf(True)
                sub_tree_node.set_category(tree_node.get_label_of_most_frequent_samples())
                tree_node.add_child_node(sub_tree_node)
                return tree_node
            else:
                _attributes_list.remove(best_attribute_to_split)
                tree_node.add_child_node(generate_decision_tree(sub_training_samples,_attributes_list))
        
        return tree_node



