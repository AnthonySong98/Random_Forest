import unittest
import numpy as np
from random_forests.utils import Dataset
from random_forests.tree_node import TreeNode
from random_forests.decision_tree import DecisionTree

class TestDecisionTree(unittest.TestCase):
    @unittest.skip("skip test_get_all_possible_values_on_attribute")
    def test_get_all_possible_values_on_attribute(self):
        test_dataset = Dataset(_dataset_name = 'watermelon_2.0', _dataset_file_path = './datasets/watermelon.csv')
        test_dataset.load_dataset(verbose=False)
        test_decision_tree = DecisionTree()
        test_decision_tree.set_training_samples_root(test_dataset.samples)
        test_decision_tree.set_attributes_list(test_dataset.feature_category_list)
        self.assertEqual(test_decision_tree.get_all_possible_values_on_attribute((0,0)),[0,1,2])
        self.assertEqual(test_decision_tree.get_all_possible_values_on_attribute((1,0)),[0,1,2])
        self.assertEqual(test_decision_tree.get_all_possible_values_on_attribute((5,0)),[0,1])

    @unittest.skip("skip test_generate_decision_tree")
    def test_generate_decision_tree(self):
        test_dataset = Dataset(_dataset_name = 'watermelon_2.0', _dataset_file_path = './datasets/watermelon.csv')
        test_dataset.load_dataset(verbose=False)
        test_decision_tree = DecisionTree()
        test_decision_tree.set_training_samples_root(test_dataset.samples)
        test_decision_tree.set_attributes_list(test_dataset.feature_category_list)
        decision_tree_root = test_decision_tree.generate_decision_tree(test_decision_tree.training_samples_root,test_decision_tree.attributes_list)
        self.assertEqual(len(decision_tree_root.child_node_list[0].child_node_list[1].child_node_list),3)
        
    @unittest.skip("skip test_generate_decision_tree_continuous")
    def test_generate_decision_tree_continuous(self):
        test_dataset = Dataset(_dataset_name = 'watermelon_3.0', _dataset_file_path = './datasets/watermelon2.csv')
        test_dataset.load_dataset(verbose=False)
        test_decision_tree = DecisionTree()
        test_decision_tree.set_training_samples_root(test_dataset.samples)
        test_decision_tree.set_attributes_list(list(range(test_dataset.num_features)))
        test_decision_tree.set_attributes_list(test_dataset.feature_category_list)
        decision_tree_root = test_decision_tree.generate_decision_tree(test_decision_tree.training_samples_root,test_decision_tree.attributes_list)

    @unittest.skip("skip test_generate_random_decision_tree_1")
    def test_generate_random_decision_tree_1(self):
        test_dataset = Dataset(_dataset_name = 'watermelon_2.0', _dataset_file_path = './datasets/watermelon.csv')
        test_dataset.load_dataset(verbose=False)
        test_decision_tree = DecisionTree()
        test_decision_tree.set_training_samples_root(test_dataset.samples)
        test_decision_tree.set_attributes_list(test_dataset.feature_category_list)
        decision_tree_root_1 = test_decision_tree.generate_decision_tree(test_decision_tree.training_samples_root,test_decision_tree.attributes_list,random_state=1)
        decision_tree_root_2 = test_decision_tree.generate_decision_tree(test_decision_tree.training_samples_root,test_decision_tree.attributes_list,random_state=2)
        
    @unittest.skip("skip test_generate_random_decision_tree_2")
    def test_generate_random_decision_tree_2(self):
        test_dataset = Dataset(_dataset_name = 'watermelon_3.0', _dataset_file_path = './datasets/watermelon2.csv')
        test_dataset.load_dataset(verbose=False)
        test_decision_tree = DecisionTree()
        test_decision_tree.set_training_samples_root(test_dataset.samples)
        test_decision_tree.set_attributes_list(test_dataset.feature_category_list)
        decision_tree_root_1 = test_decision_tree.generate_decision_tree(test_decision_tree.training_samples_root,test_decision_tree.attributes_list,random_state=1)
        decision_tree_root_2 = test_decision_tree.generate_decision_tree(test_decision_tree.training_samples_root,test_decision_tree.attributes_list,random_state=2)

    @unittest.skip("skip test_decision_tree_predict")
    def test_decision_tree_predict(self):
        test_dataset = Dataset(_dataset_name = 'watermelon_3.0', _dataset_file_path = './datasets/watermelon2.csv')
        test_dataset.load_dataset(verbose=False)
        test_decision_tree = DecisionTree()
        test_decision_tree.set_training_samples_root(test_dataset.samples)
        test_decision_tree.set_attributes_list(test_dataset.feature_category_list)
        decision_tree_root = test_decision_tree.generate_decision_tree(test_decision_tree.training_samples_root,test_decision_tree.attributes_list)
        test_decision_tree.set_root(decision_tree_root)
        for i in range((test_dataset.num_samples)):
            test_sample = test_dataset.samples[i,:]
            test_X = test_sample[0:-1]
            test_y = test_sample[-1]
            test_predicted_label = test_decision_tree.predict(test_sample=test_X)
            self.assertEqual(test_predicted_label,test_y)
        

if __name__ == '__main__':
    unittest.main()
