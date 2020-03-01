from __future__ import print_function
import pandas as pd
import numpy as np

class Dataset:
    def __init__(self, _dataset_name, _dataset_file_path):
        self.dataset_name = _dataset_name
        self.dataset_file_path = _dataset_file_path
        self.num_samples = 0
        self.num_features = 0
        self.samples = None
        self.labels = None

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
            feature_name_list = list(df.columns.values)

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

