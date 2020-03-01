import unittest
import numpy as np
from random_forests.utils import Dataset
from random_forests.tree_node import TreeNode
from random_forests.decision_tree import DecisionTree

class TestDecisionTree(unittest.TestCase):
    def test_get_all_possible_values_on_attribute(self):
        test_dataset = Dataset(_dataset_name = 'watermelon_2.0', _dataset_file_path = './datasets/watermelon.csv')
        test_dataset.load_dataset(verbose=False)
        test_decision_tree = DecisionTree()
        test_decision_tree.set_training_samples_root(test_dataset.samples)
        test_decision_tree.set_attributes_list(list(range(test_dataset.num_features)))
        self.assertEqual(test_decision_tree.get_all_possible_values_on_attribute(0),[0,1,2])
        self.assertEqual(test_decision_tree.get_all_possible_values_on_attribute(1),[0,1,2])
        self.assertEqual(test_decision_tree.get_all_possible_values_on_attribute(5),[0,1])

    def test_generate_decision_tree(self):
        test_dataset = Dataset(_dataset_name = 'watermelon_2.0', _dataset_file_path = './datasets/watermelon.csv')
        test_dataset.load_dataset(verbose=False)
        test_decision_tree = DecisionTree()
        test_decision_tree.set_training_samples_root(test_dataset.samples)
        test_decision_tree.set_attributes_list(list(range(test_dataset.num_features)))
        decision_tree_root = test_decision_tree.generate_decision_tree(test_decision_tree.training_samples_root,test_decision_tree.attributes_list)
        self.assertEqual(len(decision_tree_root.child_node_list[0].child_node_list[1].child_node_list),3)
        





if __name__ == '__main__':
    unittest.main()
