from random import randrange
from random_forests.decision_tree import DecisionTree
from random_forests.utils import Dataset , VisTree

class RandomForest():
    def __init__(self,n_estimators):
        self.dataset = None
        self.forest = []
        self.n_estimators = n_estimators

    def set_dataset(self,dataset):
        self.dataset = dataset

    def bootstrap_aggregating(self):
        random_selected_training_samples_list = []
        for i in range(self.n_estimators):
            random_selected_samples_index_list = [ randrange(self.dataset.num_samples) for i in range(self.dataset.num_samples)]
            random_selected_training_samples_list.append(self.dataset.samples[random_selected_samples_index_list,:])
        return random_selected_training_samples_list

    def generate_random_forest(self):
        random_selected_training_samples_list = self.bootstrap_aggregating()
        for random_selected_training_samples in random_selected_training_samples_list:
            test_decision_tree = DecisionTree()
            test_decision_tree.set_training_samples_root(random_selected_training_samples)
            test_decision_tree.set_attributes_list(self.dataset.feature_category_list)
            decision_tree_root = test_decision_tree.generate_decision_tree(test_decision_tree.training_samples_root,test_decision_tree.attributes_list,random_state=1)
            test_decision_tree.set_root(decision_tree_root)
            self.forest.append(test_decision_tree)


