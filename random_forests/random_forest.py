import numpy as np
from random import randrange
from collections import Counter 
from random_forests.decision_tree import DecisionTree
from random_forests.utils import Dataset , VisTree

class RandomForest():
    def __init__(self,n_estimators,n_samples = 100):
        self.dataset = None
        self.forest = []
        self.n_estimators = n_estimators
        self.n_samples = n_samples
        self.out_of_bag_testing_samples_dict = {}

    def set_dataset(self,dataset):
        self.dataset = dataset

    def bootstrap_aggregating(self):
        random_selected_training_samples_list = []
        for i in range(self.n_estimators):
            random_selected_samples_index_list = [ randrange(self.dataset.num_samples) for i in range(self.n_samples)]
            random_selected_training_samples_list.append(self.dataset.samples[random_selected_samples_index_list,:])
            out_of_bag_samples_index_list = list(set(list(range(self.dataset.num_samples))).difference(set(random_selected_samples_index_list)))
            for out_of_bag_samples_index in out_of_bag_samples_index_list:
                if out_of_bag_samples_index not in self.out_of_bag_testing_samples_dict:
                    self.out_of_bag_testing_samples_dict[out_of_bag_samples_index] = [i]
                else:
                    self.out_of_bag_testing_samples_dict[out_of_bag_samples_index].append(i)
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

    def calculate_out_of_bag_error(self):
        if len(self.forest) != self.n_estimators:
            print("Random Forest genertion error!")
            return
        error_cnt = 0
        for sample_index in self.out_of_bag_testing_samples_dict:
            test_sample = self.dataset.samples[sample_index,:]
            predicted_label_vote = []
            for decision_tree_index in self.out_of_bag_testing_samples_dict[sample_index]:
                predicted_label_vote.append(self.forest[decision_tree_index].predict(test_sample[:-1]))
            predicted_label = Counter(predicted_label_vote).most_common(1)[0][0]
            if predicted_label != test_sample[-1]:
                error_cnt += 1
        
        return  float (error_cnt) / len(self.out_of_bag_testing_samples_dict)

                

    def predict(self,test_sample):
        vote = []
        if len(self.forest) != self.n_estimators:
            print("Random Forest genertion error!")
            return
        for decision_tree in self.forest:
            vote.append(decision_tree.predict(test_sample))
        return Counter(vote).most_common(1)[0][0]

    def predict_batch(self,test_samples):
        predicted_result = []
        for i in range(test_samples.shape[0]):
            test_sample = test_samples[i,:]
            predicted_result.append(self.predict(test_sample))
        return predicted_result

    def get_accuracy(self,test_samples):
        actual_labels = (test_samples[:,-1])
        predicted_labels = np.asarray(self.predict_batch(test_samples))
        accuracy = 1 - float (np.sum(np.abs(actual_labels - predicted_labels))) / actual_labels.shape[0]
        return accuracy
        
