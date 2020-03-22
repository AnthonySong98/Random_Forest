import numpy as np
from random_forests.tree_node import TreeNode

class DecisionTree:
    def __init__(self):
        self.root = TreeNode()
        self.training_samples_root = None
        self.attributes_list = []
        
    def set_root(self,_tree_node):
        self.root = _tree_node

    def set_training_samples_root(self,_training_samples):
        self.training_samples_root = _training_samples

    def set_attributes_list(self,_attributes_list):
        self.attributes_list = _attributes_list

    def get_all_possible_values_on_attribute(self,_target_attribute):
        if _target_attribute[1] == 0:
            return list(set(self.training_samples_root[:,_target_attribute[0]].tolist()))

    def generate_decision_tree(self,_training_samples, _attributes_list, random_state = None):
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

        best_attribute_to_split = tree_node.select_best_attribute_to_split(max_features=random_state)

        # set the target split attribute
        tree_node.set_target_attribute(best_attribute_to_split)

        if best_attribute_to_split[1] == 0:
            attribute_values_samples_mapping_dict = tree_node.split_by_attribute_internal(best_attribute_to_split)
            for possible_value in self.get_all_possible_values_on_attribute(best_attribute_to_split):
                sub_tree_node = TreeNode()
                #sub_training_samples = tree_node.split_by_attribute(possible_value)
                if possible_value in attribute_values_samples_mapping_dict:
                    sub_training_samples = attribute_values_samples_mapping_dict[possible_value]
                else:
                    sub_training_samples = np.zeros((0,0))

                if sub_training_samples.shape[0] == 0:
                    sub_tree_node.set_samples(sub_training_samples)
                    sub_tree_node.set_is_leaf(True)
                    sub_tree_node.set_category(tree_node.get_label_of_most_frequent_samples())
                    tree_node.add_child_node_criterion(possible_value)
                    tree_node.add_child_node(sub_tree_node)
                    return tree_node
                else:
                    _attributes_list_cp = _attributes_list[:]
                    _attributes_list_cp.remove(best_attribute_to_split)
                    tree_node.add_child_node_criterion(possible_value)
                    tree_node.add_child_node(self.generate_decision_tree(sub_training_samples,_attributes_list_cp,random_state = random_state))
        
        else:
            attribute_values_samples_mapping_dict = tree_node.split_by_attribute_internal(best_attribute_to_split)
            for possible_criterion_value in attribute_values_samples_mapping_dict:
                sub_tree_node = TreeNode()

                sub_training_samples = attribute_values_samples_mapping_dict[possible_criterion_value]

                if sub_training_samples.shape[0] == 0:
                    return None
                else:
                    _attributes_list_cp = _attributes_list[:]
                    _attributes_list_cp.remove(best_attribute_to_split)
                    tree_node.add_child_node_criterion(possible_criterion_value)
                    tree_node.add_child_node(self.generate_decision_tree(sub_training_samples,_attributes_list_cp,random_state = random_state))

        return tree_node


    def predict(self, test_sample):
        current_tree_node = self.root
        while not current_tree_node.is_leaf:
            targeted_attribute =  test_sample[[current_tree_node.target_attribute[0]]]
            child_node_index = -1
            for current_criterion in current_tree_node.child_node_criterion_list:
                if current_tree_node.target_attribute[1] == 0:
                    if current_criterion == targeted_attribute:
                        child_node_index = current_tree_node.child_node_criterion_list.index(current_criterion)
                        break
                else:
                    if current_criterion[:2] == "<=" and targeted_attribute <= float(current_criterion[2:]):
                        child_node_index = current_tree_node.child_node_criterion_list.index(current_criterion)
                        break
                    elif current_criterion[:1] == ">" and targeted_attribute > float(current_criterion[1:]):
                        child_node_index = current_tree_node.child_node_criterion_list.index(current_criterion)
                        break
            if child_node_index != -1:
                current_tree_node = current_tree_node.child_node_list[child_node_index]
            else:
                print("No satisfied criterion is found in tree node!")
        return current_tree_node.category



