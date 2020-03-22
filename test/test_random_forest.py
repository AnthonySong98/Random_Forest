import unittest

from random_forests.random_forest import RandomForest
from random_forests.utils import Dataset,VisTree

class TestRandomForest(unittest.TestCase):
    @unittest.skip("skip test_random_forest_1")
    def test_random_forest_1(self):
        test_dataset = Dataset(_dataset_name = 'watermelon_2.0', _dataset_file_path = './datasets/watermelon.csv')
        test_dataset.load_dataset(verbose=False)
        
        test_random_forest = RandomForest(10)
        test_random_forest.set_dataset(test_dataset)
        test_random_forest.generate_random_forest()
        i = 0
        for tree in test_random_forest.forest:
            test_vis_tree = VisTree(tree,test_dataset.feature2number_mapping,\
                test_dataset.feature_name_list,_tree_name="test_1_random_decision_tree_%d" %(i))
            test_vis_tree.vis_tree(mode=1)
            i += 1


    @unittest.skip("skip test_random_forest_2")
    def test_random_forest_2(self):
        test_dataset = Dataset(_dataset_name = 'watermelon_3.0', _dataset_file_path = './datasets/watermelon2.csv')
        test_dataset.load_dataset(verbose=False)
        
        test_random_forest = RandomForest(10)
        test_random_forest.set_dataset(test_dataset)
        test_random_forest.generate_random_forest()
        i = 0
        for tree in test_random_forest.forest:
            test_vis_tree = VisTree(tree,test_dataset.feature2number_mapping,\
                test_dataset.feature_name_list,_tree_name="test_2_random_decision_tree_%d" %(i))
            test_vis_tree.vis_tree(mode=1)
            i += 1

if __name__ == '__main__':
    unittest.main()