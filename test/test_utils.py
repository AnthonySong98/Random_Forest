import unittest

from random_forests.decision_tree import DecisionTree
from random_forests.utils import VisTree,Dataset

class TestVisTree(unittest.TestCase):
    def test_vis_tree(self):
        test_dataset = Dataset(_dataset_name = 'watermelon_2.0', _dataset_file_path = './datasets/watermelon.csv')
        test_dataset.load_dataset(verbose=False)
        test_decision_tree = DecisionTree()
        test_decision_tree.set_training_samples_root(test_dataset.samples)
        test_decision_tree.set_attributes_list(list(range(test_dataset.num_features)))
        decision_tree_root = test_decision_tree.generate_decision_tree(test_decision_tree.training_samples_root,test_decision_tree.attributes_list)
        test_decision_tree.set_root(decision_tree_root)
        test_vis_tree = VisTree(test_decision_tree,test_dataset.feature2number_mapping,test_dataset.feature_name_list)
        test_vis_tree.vis_tree()
        self.assertEqual(1,2-1)

if __name__ == '__main__':
    unittest.main()