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
        # i = 0
        # for tree in test_random_forest.forest:
        #     test_vis_tree = VisTree(tree,test_dataset.feature2number_mapping,\
        #         test_dataset.feature_name_list,_tree_name="test_1_random_decision_tree_%d" %(i))
        #     test_vis_tree.vis_tree(mode=1)
        #     i += 1


    @unittest.skip("skip test_random_forest_2")
    def test_random_forest_2(self):
        test_dataset = Dataset(_dataset_name = 'watermelon_3.0', _dataset_file_path = './datasets/watermelon2.csv')
        test_dataset.load_dataset(verbose=False)
        
        test_random_forest = RandomForest(10)
        test_random_forest.set_dataset(test_dataset)
        test_random_forest.generate_random_forest()
        i = 0
        # for tree in test_random_forest.forest:
        #     test_vis_tree = VisTree(tree,test_dataset.feature2number_mapping,\
        #         test_dataset.feature_name_list,_tree_name="test_2_random_decision_tree_%d" %(i))
        #     test_vis_tree.vis_tree(mode=1)
        #     i += 1

    # @unittest.skip("skip test_random_forest_3")
    def test_random_forest_3(self):
        test_dataset = Dataset(_dataset_name = 'uci_blood', _dataset_file_path = './datasets/uci_blood.csv')
        test_dataset.load_dataset(verbose=False)
        
        test_random_forest = RandomForest(n_estimators = 20,n_samples=400)
        test_random_forest.set_dataset(test_dataset)
        import time 
        start = time.clock()
        test_random_forest.generate_random_forest()
        end = time.clock()
        print((end-start)/20.0)
        """ i = 0
        for tree in test_random_forest.forest:
            test_vis_tree = VisTree(tree,test_dataset.feature2number_mapping,\
                test_dataset.feature_name_list,_tree_name="test_3_random_decision_tree_%d" %(i))
            test_vis_tree.vis_tree(mode=1)
            i += 1
 """
        print(test_random_forest.calculate_out_of_bag_error())

    @unittest.skip("skip test_random_forest_predict")
    def test_random_forest_predict(self):
        test_dataset = Dataset(_dataset_name = 'watermelon_3.0', _dataset_file_path = './datasets/watermelon2.csv')
        test_dataset.load_dataset(verbose=False)
        
        test_random_forest = RandomForest(10)
        test_random_forest.set_dataset(test_dataset)
        test_random_forest.generate_random_forest()

        # i = 0
        # for tree in test_random_forest.forest:
        #     test_vis_tree = VisTree(tree,test_dataset.feature2number_mapping,\
        #         test_dataset.feature_name_list,_tree_name="test_2_random_decision_tree_%d" %(i))
        #     test_vis_tree.vis_tree(mode=1)
        #     i += 1

        for i in range((test_dataset.num_samples)):
            test_sample = test_dataset.samples[i,:]
            test_X = test_sample[0:-1]
            test_y = test_sample[-1]
            test_predicted_label = test_random_forest.predict(test_sample=test_X)

    @unittest.skip("skip test_predict_batch")
    def test_predict_batch(self):
        test_dataset = Dataset(_dataset_name = 'watermelon_3.0', _dataset_file_path = './datasets/watermelon2.csv')
        test_dataset.load_dataset(verbose=False)
        
        test_random_forest = RandomForest(10)
        test_random_forest.set_dataset(test_dataset)
        test_random_forest.generate_random_forest()

        predicted_result = test_random_forest.predict_batch(test_dataset.samples)

    @unittest.skip("skip test_get_accuracy")
    def test_get_accuracy(self):
        test_dataset = Dataset(_dataset_name = 'watermelon_3.0', _dataset_file_path = './datasets/watermelon2.csv')
        test_dataset.load_dataset(verbose=False)
        
        test_random_forest = RandomForest(10)
        test_random_forest.set_dataset(test_dataset)
        test_random_forest.generate_random_forest()

        (test_random_forest.get_accuracy(test_dataset.samples))


if __name__ == '__main__':
    unittest.main()