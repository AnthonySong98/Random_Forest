import unittest
import numpy as np
from random_forests.utils import Dataset
from random_forests.tree_node import TreeNode

class TestTreeNode(unittest.TestCase):
    def test_set_samples(self):
        test_tree_node = TreeNode()
        test_tree_node.set_samples(np.zeros((4,5)))
        self.assertEqual(test_tree_node.samples.shape,np.zeros((4,5)).shape)

    def test_is_belong_to_same_category(self):
        test_tree_node = TreeNode()

        # test all false labels
        test_tree_node.set_samples(np.zeros((4,5)))
        test_result,test_result_label = test_tree_node.is_belong_to_same_category()
        self.assertEqual(test_result,True)
        self.assertEqual(test_result_label,0)

        # test all true labels
        test_tree_node.set_samples(np.ones((4,5)))
        test_result,test_result_label = test_tree_node.is_belong_to_same_category()
        self.assertEqual(test_result,True)
        self.assertEqual(test_result_label,1)

        # test some true labels and false labels
        input_np = np.random.randint(2, size=(5, 4))
        input_np[0][3] = 1
        input_np[1][3] = 0
        test_tree_node.set_samples(input_np)
        test_result,test_result_label = test_tree_node.is_belong_to_same_category()
        self.assertEqual(test_result,False)
        self.assertEqual(test_result_label,None)

    def test_is_have_same_value_on_attribute_list(self):
        # test when return true
        test_tree_node = TreeNode()
        input_np = np.ones((4,5))
        input_np[0][2] = 0
        test_tree_node.set_samples(input_np)
        test_tree_node.set_attribute_list([0,1,3])
        self.assertEqual(test_tree_node.is_have_same_value_on_attribute_list(),True)

        # test when return false
        test_tree_node.set_attribute_list([0,2,3])
        self.assertEqual(test_tree_node.is_have_same_value_on_attribute_list(),False)

    def test_get_label_of_most_frequent_samples(self):
        test_tree_node = TreeNode()
        input_np = np.ones((4,5))
        input_np[0][4] = 0
        input_np[1][4] = 0
        # input_np[2][4] = 0
        test_tree_node.set_samples(input_np)
        self.assertEqual(test_tree_node.get_label_of_most_frequent_samples(),0)

    def test_get_ent(self):
        test_tree_node = TreeNode()
        ent1 = test_tree_node.get_ent(np.ones((4,5)))
        self.assertEqual(ent1,0)

        ent2 = test_tree_node.get_ent(np.zeros((4,5)))
        self.assertEqual(ent2,0)

        input_np = np.ones((4,5))
        input_np[0][4] = 0
        input_np[1][4] = 0
        ent2 = test_tree_node.get_ent(input_np)
        self.assertEqual(ent2,1)

        test_dataset = Dataset(_dataset_name = 'watermelon_2.0', _dataset_file_path = './datasets/watermelon.csv')
        test_dataset.load_dataset(verbose=False)
        ent3 = test_tree_node.get_ent(test_dataset.samples)
        self.assertAlmostEqual(ent3,0.9975025463691152)

    def test_split_by_attribute_internal(self):
        test_tree_node = TreeNode()
        test_dataset = Dataset(_dataset_name = 'watermelon_2.0', _dataset_file_path = './datasets/watermelon.csv')
        test_dataset.load_dataset(verbose=False)
        test_tree_node.set_samples(test_dataset.samples)
        test_tree_node.set_attribute_list(list(range(6)))
        attribute_values_samples_mapping_dict = test_tree_node.split_by_attribute_internal(3)
        self.assertEqual(attribute_values_samples_mapping_dict[0].shape[0],9)
        self.assertEqual(attribute_values_samples_mapping_dict[1].shape[0],5)
        self.assertEqual(attribute_values_samples_mapping_dict[2].shape[0],3)

    def test_select_best_attribute_to_split(self):
        test_tree_node = TreeNode()
        test_dataset = Dataset(_dataset_name = 'watermelon_2.0', _dataset_file_path = './datasets/watermelon.csv')
        test_dataset.load_dataset(verbose=False)
        test_tree_node.set_samples(test_dataset.samples)
        test_tree_node.set_attribute_list(list(range(6)))
        best_attribute_to_split = test_tree_node.select_best_attribute_to_split()
        self.assertEqual(best_attribute_to_split,3)


if __name__ == '__main__':
    unittest.main()