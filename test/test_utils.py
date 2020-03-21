import unittest

from random_forests.decision_tree import DecisionTree
from random_forests.utils import VisTree,Dataset

class TestVisTree(unittest.TestCase):
    @unittest.skip("skip test_vis_tree_1")
    def test_vis_tree_1(self):
        test_dataset = Dataset(_dataset_name = 'watermelon_2.0', _dataset_file_path = './datasets/watermelon.csv')
        test_dataset.load_dataset(verbose=False)
        test_decision_tree = DecisionTree()
        test_decision_tree.set_training_samples_root(test_dataset.samples)
        test_decision_tree.set_attributes_list([(0,0),(1,0),(2,0),(3,0),(4,0),(5,0)])
        decision_tree_root = test_decision_tree.generate_decision_tree(test_decision_tree.training_samples_root,test_decision_tree.attributes_list)
        test_decision_tree.set_root(decision_tree_root)
        test_vis_tree = VisTree(test_decision_tree,test_dataset.feature2number_mapping,\
            test_dataset.feature_name_list,_tree_name="test_decision_tree")
        test_vis_tree.vis_tree(mode=1)

    @unittest.skip("skip test_vis_tree_2")
    def test_vis_tree_2(self):
        test_dataset = Dataset(_dataset_name = 'watermelon_3.0', _dataset_file_path = './datasets/watermelon2.csv')
        test_dataset.load_dataset(verbose=False)
        test_decision_tree = DecisionTree()
        test_decision_tree.set_training_samples_root(test_dataset.samples)
        test_decision_tree.set_attributes_list(list(range(test_dataset.num_features)))
        test_decision_tree.set_attributes_list([(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,1),(7,1)])
        decision_tree_root = test_decision_tree.generate_decision_tree(test_decision_tree.training_samples_root,test_decision_tree.attributes_list)
        test_decision_tree.set_root(decision_tree_root)
        test_vis_tree = VisTree(test_decision_tree,test_dataset.feature2number_mapping,\
            test_dataset.feature_name_list,_tree_name="test_decision_tree_2")
        test_vis_tree.vis_tree(mode=1)

    @unittest.skip("skip test_vis_tree_3")
    def test_vis_tree_3(self):
        test_dataset = Dataset(_dataset_name = 'tennis', _dataset_file_path = './datasets/tennis.csv')
        test_dataset.load_dataset(verbose=False)
        test_decision_tree = DecisionTree()
        test_decision_tree.set_training_samples_root(test_dataset.samples)
        test_decision_tree.set_attributes_list(list(range(test_dataset.num_features)))
        test_decision_tree.set_attributes_list([(0,0),(1,0),(2,0),(3,0)])
        decision_tree_root = test_decision_tree.generate_decision_tree(test_decision_tree.training_samples_root,test_decision_tree.attributes_list)
        test_decision_tree.set_root(decision_tree_root)
        test_vis_tree = VisTree(test_decision_tree,test_dataset.feature2number_mapping,\
            test_dataset.feature_name_list,_tree_name="test_decision_tree_3")
        test_vis_tree.vis_tree(mode=1)

    # @unittest.skip("skip test_vis_tree_4")
    def test_vis_tree_4(self):
        test_dataset = Dataset(_dataset_name = 'watermelon_2.0', _dataset_file_path = './datasets/watermelon.csv')
        test_dataset.load_dataset(verbose=False)
        for i in range(10):
            test_decision_tree = DecisionTree()
            test_decision_tree.set_training_samples_root(test_dataset.samples)
            test_decision_tree.set_attributes_list([(0,0),(1,0),(2,0),(3,0),(4,0),(5,0)])
            decision_tree_root = test_decision_tree.generate_decision_tree(test_decision_tree.training_samples_root,test_decision_tree.attributes_list,\
                random_state=1)
            test_decision_tree.set_root(decision_tree_root)
            test_vis_tree = VisTree(test_decision_tree,test_dataset.feature2number_mapping,\
                test_dataset.feature_name_list,_tree_name="test_random_decision_tree_%d" %(i))
            test_vis_tree.vis_tree(mode=1)

if __name__ == '__main__':
    unittest.main()