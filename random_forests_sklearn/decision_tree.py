from sklearn import tree
from sklearn.datasets import load_iris

# X, y = load_iris(return_X_y=True)
# print(X)
# print(y)
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, y)
# tree.plot_tree(clf.fit(X, y)) 

import graphviz 
# dot_data = tree.export_graphviz(clf, out_file="./random_forests_sklearn/vis/iris") 
# graph = graphviz.Source(dot_data)
# graphviz.render(engine="dot",format="pdf",filepath="./random_forests_sklearn/vis/iris") 

from random_forests.utils import Dataset
test_dataset = Dataset(_dataset_name = 'watermelon_2.0', _dataset_file_path = './datasets/watermelon.csv')
test_dataset.load_dataset(verbose=False)
X = test_dataset.samples[:,:-1].astype(int)
y = test_dataset.labels.astype(int)
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X,y)

dot_data = tree.export_graphviz(clf, out_file="./random_forests_sklearn/vis/watermelon.dot") 
graph = graphviz.Source(dot_data)
graphviz.render(engine="dot",format="pdf",filepath="./random_forests_sklearn/vis/watermelon.dot") 

