from __future__ import print_function
import pandas as pd
import numpy as np
import os
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

        if self.dataset_name == 'watermelon_3.0':
            df = pd.read_csv(filepath_or_buffer=self.dataset_file_path,sep=',')
            if verbose:
                print(df)
            feature2number_mapping = [{'Green':0,'Black':1,'White':2},\
                                        {'Curl':0,'Roll':1,'Stiff':2},\
                                        {'Dull':0,'Depressing':1,'Crispy':2},\
                                        {'Clear':0,'Fuzzy':1,'Blurry':2},\
                                        {'Concave':0,'Hollow':1,'Flat':2},\
                                        {'Smooth':0,'Stick':1},\
                                        {},\
                                        {}]
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
        
        if self.dataset_name == 'tennis':
            df = pd.read_csv(filepath_or_buffer=self.dataset_file_path,sep=',')
            if verbose:
                print(df)
            feature2number_mapping = [{'Sunny':0,'Overcast':1,'Rainy':2},\
                                        {'Hot':0,'Mild':1,'Cool':2},\
                                        {'High':0,'Normal':1},\
                                        {'False':0,'True':1},\
                                        ]
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

                if feature_name == 'PlayTennis':
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
    def __init__(self,_decision_tree, _feature2number_mapping, _feature_name_list, _tree_name):
        self.tree_root = _decision_tree.root
        self.number2feature_mapping = [dict(zip(i_dict.values(),i_dict.keys())) for i_dict in _feature2number_mapping]
        self.feature_name_list = _feature_name_list
        self.tree_name = _tree_name
    
    def print_node(self,_temp_node):
        '''
        print node info to console
        '''
        print("leaf ? ",_temp_node.is_leaf)
        # print("attribute_list : ",[self.feature_name_list[i] for i in _temp_node.attribute_list])
        # print("target_attribute : ",self.feature_name_list[_temp_node.target_attribute])
        # print("child_node_criterion_list : ",[self.number2feature_mapping[_temp_node.target_attribute][i] for i in _temp_node.child_node_criterion_list])
        print("samples : ",_temp_node.samples.shape)
        if _temp_node.is_leaf == True:
            print("Category : ", _temp_node.category)
        else:
            print("attribute_list : ",[self.feature_name_list[i] for i in _temp_node.attribute_list])
            print("target_attribute : ",self.feature_name_list[_temp_node.target_attribute])
            print("child_node_criterion_list : ",[self.number2feature_mapping[_temp_node.target_attribute][i] for i in _temp_node.child_node_criterion_list])

    def dot4node(self,_temp_node,_node_idx):
        '''
        generate dot file for node
        '''
        if _temp_node.is_leaf:
            return ("node_%d [shape = ellipse,label= \"category: %s\\nsamples: %d\"];\n" %(_node_idx,_temp_node.category,(_temp_node.samples.shape[0])))
        else:
            return ("node_%d [shape = box,label= \"target attribute: %s ?\\nsamples: %d\"];\n"%(_node_idx,self.feature_name_list[_temp_node.target_attribute[0]],(_temp_node.samples.shape[0])))

    def dot4edge(self,node1_idx,node2_idx,label_content):
        '''
        generate dot file for edge from node1 to node2 with label_content
        '''
        return ("node_%d -> node_%d [label=\"%s\"];\n"%(node1_idx,node2_idx,str(label_content)))

    def vis_tree(self,mode = 0):
        if mode == 0:

            helper_queue = []

            current_node_idx = 0
            helper_queue.append((self.tree_root,current_node_idx))

            print("Node ",current_node_idx)
            self.print_node(self.tree_root)

            node_idx = 0
            while len(helper_queue) > 0:
                print('==============================================================')

                temp_node,current_node_idx = helper_queue.pop(0)

                current_node_child_id = 0
                for child_node in temp_node.child_node_list:
                    node_idx += 1
                    print()
                    print("Node ",node_idx)
                    self.print_node(child_node)
                    helper_queue.append((child_node,node_idx))

                    print("Node ", current_node_idx , " ---> " ,node_idx)
                    print("Attribute vaule ", self.number2feature_mapping[temp_node.target_attribute][temp_node.child_node_criterion_list[current_node_child_id]])
                    current_node_child_id += 1

        if mode == 1 :

            f = open("./vis/dot_files/%s.gv" %(self.tree_name),"w")

            f.write("digraph %s {\n" %(self.tree_name))
            helper_queue = []

            current_node_idx = 0
            helper_queue.append((self.tree_root,current_node_idx))

            f.write(self.dot4node(self.tree_root,current_node_idx))

            node_idx = 0
            while len(helper_queue) > 0:

                temp_node,current_node_idx = helper_queue.pop(0)

                current_node_child_id = 0
                for child_node in temp_node.child_node_list:
                    node_idx += 1

                    f.write(self.dot4node(child_node,node_idx))
                    helper_queue.append((child_node,node_idx))
                    if temp_node.target_attribute[1] == 0:
                        f.write(self.dot4edge(current_node_idx,node_idx,\
                            str(self.number2feature_mapping[temp_node.target_attribute[0]][temp_node.child_node_criterion_list[current_node_child_id]])))
                    else:
                        f.write(self.dot4edge(current_node_idx,node_idx,\
                            str(temp_node.child_node_criterion_list[current_node_child_id])))

                    current_node_child_id += 1

            f.write("}\n")
            f.close()

            os.system("dot -Tpdf ./vis/dot_files/%s.gv -o ./vis/pdf_files/%s.pdf" %(self.tree_name,self.tree_name))

