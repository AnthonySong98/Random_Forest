from __future__ import print_function
import pandas as pd
import numpy as np

from random_forests.decision_tree import DecisionTree
from random_forests.tree_node import TreeNode

class Dataset:
    def __init__(self, _dataset_name, _dataset_file_path):
        self.dataset_name = _dataset_name
        self.dataset_file_path = _dataset_file_path
        self.num_samples = 0
        self.num_features = 0
        self.samples = None
        self.labels = None
        self.feature2number_mapping = []
        self.feature_name_list = []

    def load_dataset(self,verbose = True):
        if self.dataset_name == 'watermelon_2.0':
            df = pd.read_csv(filepath_or_buffer=self.dataset_file_path,sep=',')
            if verbose:
                print(df)
            feature2number_mapping = [{'Green':0,'Black':1,'White':2},\
                                        {'Curl':0,'Roll':1,'Stiff':2},\
                                        {'Dull':0,'Depressing':1,'Crispy':2},\
                                        {'Clear':0,'Fuzzy':1,'Blurry':2},\
                                        {'Concave':0,'Hollow':1,'Flat':2},\
                                        {'Smooth':0,'Stick':1}]
            self.feature2number_mapping = feature2number_mapping

            feature_name_list = list(df.columns.values)
            self.feature_name_list = feature_name_list

            feature_num = len(feature_name_list) - 1
            self.num_features = feature_num

            sample_num = df.shape[0]
            self.num_samples = sample_num

            np_dataset = np.zeros((df.shape[0],df.shape[1]))
            cnt = 0
            for feature_name in feature_name_list:
                feature_col_np = (df[feature_name].to_numpy())

                if feature_name == 'Result':
                    for i in range(sample_num):
                        np_dataset[i][cnt] = feature_col_np[i]
                    break
                
                for i in range(sample_num):
                    if feature_col_np[i] in feature2number_mapping[cnt]:
                        np_dataset[i][cnt] = (feature2number_mapping[cnt][feature_col_np[i]])
                    else:
                        np_dataset[i][cnt] = feature_col_np[i]
                cnt += 1
            if verbose:
                print(np_dataset)
            self.samples = np_dataset
            self.labels = (np_dataset[:,-1])


class VisTree:
    def __init__(self,_decision_tree, _feature2number_mapping, _feature_name_list):
        self.tree_root = _decision_tree.root
        self.number2feature_mapping = [dict(zip(i_dict.values(),i_dict.keys())) for i_dict in _feature2number_mapping]
        self.feature_name_list = _feature_name_list
    
    def print_node(self,_temp_node):
        print("leaf ? ",_temp_node.is_leaf)
        print("attribute_list : ",[self.feature_name_list[i] for i in _temp_node.attribute_list])
        print("target_attribute : ",self.feature_name_list[_temp_node.target_attribute])
        print("child_node_criterion_list : ",[self.number2feature_mapping[_temp_node.target_attribute][i] for i in _temp_node.child_node_criterion_list])
        print("samples : ",_temp_node.samples.shape)
        if _temp_node.is_leaf == True:
            print("Category : ", _temp_node.category)


    def vis_tree(self):
        helper_queue = []
        temp_node = self.tree_root
        while temp_node is not None:
            print('==============================================================')
            self.print_node(temp_node)
            # print("leaf ? ",temp_node.is_leaf)
            # print("attribute_list : ",temp_node.attribute_list)
            # print("target_attribute : ",temp_node.target_attribute)
            # print("child_node_criterion_list : ",temp_node.child_node_criterion_list)
            # print("samples : ",temp_node.samples.shape)
            # if temp_node.is_leaf == True:
            #     print("Category : ", temp_node.category)
            
            for child_node in temp_node.child_node_list:
                helper_queue.append(child_node)

            if len(helper_queue) > 0 :
                temp_node = helper_queue.pop(0)
            else:
                temp_node = None



