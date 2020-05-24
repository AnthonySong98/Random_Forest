import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
# from random_forests.utils import Dataset
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve
sns.set_style('darkgrid')
# test_dataset = Dataset(_dataset_name = 'uci_blood', _dataset_file_path = './datasets/uci_blood.csv')
# test_dataset.load_dataset(verbose=False)
# X = test_dataset.samples[:,:-1].astype(int)
# y = test_dataset.labels.astype(int)
# clf = RandomForestClassifier(n_estimators = 100, criterion="entropy",oob_score=True)
# clf = clf.fit(X,y)
# print(clf.oob_score_)

def vis_box_plot():
    dataset_root_path = "datasets/blood_donation"

    total_X = None
    total_y = None
    cnt = 0

    for root, dirs, files in os.walk(dataset_root_path):
        for file in files:
            dataset_path = (os.path.join(root, file))
            dataset_name = os.path.basename(dataset_path)[:-4]
            print(dataset_path)
            print(dataset_name)

            raw_data = np.load(dataset_path)
            X = raw_data['arr_0']
            y = raw_data['arr_1']

            if cnt == 0:
                total_X = X
                total_y = y
            else:
                total_X = np.concatenate((total_X,X),axis=0)
                total_y = np.concatenate((total_y,y))
            
            cnt += 1
    
    X_train, X_test, y_train, y_test = train_test_split(total_X, total_y, random_state=42)

    map_list = {0:'Age',1:'Sex',2:'Last donation volume',3:'Total donation volume',4:'Total number of donation',
    5:'Last time eligibility',6:'Occupation',7:'Interval',8:'Education',
    9:'Neighbourhood',10:'Reaction',11:'Frequency'}
    for i in range(12):

        fig1, ax1 = plt.subplots()
        # Create the boxplot
        ax1.set_xticklabels([map_list[i]])
        bp = ax1.boxplot(total_X[:,i],patch_artist=True)
        for box in bp['boxes']:
            # change outline color
            box.set( color='#7570b3', linewidth=1)
            # change fill color
            box.set( facecolor = '#1b9e77' )

        ## change color and linewidth of the whiskers
        for whisker in bp['whiskers']:
            whisker.set(color='#7570b3', linewidth=1)

        ## change color and linewidth of the caps
        for cap in bp['caps']:
            cap.set(color='#7570b3', linewidth=1)

        ## change color and linewidth of the medians
        for median in bp['medians']:
            median.set(color='#b2df8a', linewidth=1)

        ## change the style of fliers and their fill
        for flier in bp['fliers']:
            flier.set(marker='o', color='#e7298a', alpha=0.25)

        ax1.set_title('Box plot for attribute ' + map_list[i])

        fig1.savefig('random_forests_sklearn/vis/'+ map_list[i] +'.png', bbox_inches='tight')
        fig1.clf()

        fig2, ax2 = plt.subplots()
        
        dp = sns.distplot(total_X[:,i])#, hist=True, kde=False)
        fig2.suptitle('Histogram for attribute ' + map_list[i])
        ax2.set_xlabel(map_list[i])
        ax2.set_ylabel('Density')   # relative to plt.rcParams['font.size']
        fig2.savefig('random_forests_sklearn/vis/dis_'+ map_list[i] +'.png')
        fig2.clf()



def main():
    # tune_hyper_parameter()
    vis_box_plot()


if __name__ == "__main__":
    main()

# random_forests_sklearn.data_vis