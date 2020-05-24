from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier
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
# test_dataset = Dataset(_dataset_name = 'watermelon_2.0', _dataset_file_path = './datasets/watermelon.csv')
# test_dataset.load_dataset(verbose=False)
# X = test_dataset.samples[:,:-1].astype(int)
# y = test_dataset.labels.astype(int)
# clf = tree.DecisionTreeClassifier(criterion="entropy")
# clf = clf.fit(X,y)

test_dataset = Dataset(_dataset_name = 'uci_blood', _dataset_file_path = './datasets/uci_blood.csv')
test_dataset.load_dataset(verbose=False)
X = test_dataset.samples[:,:-1].astype(int)
y = test_dataset.labels.astype(int)
clf = tree.DecisionTreeClassifier(criterion="entropy")
stupid_clf = DummyClassifier(strategy='uniform')
RF_clf = RandomForestClassifier(n_estimators = 200)

clf = clf.fit(X,y)
RF_clf = RF_clf.fit(X,y)

aver_list = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
aver_list2 = cross_val_score(stupid_clf,X, y, cv=5, scoring='accuracy')
aver_list3 = cross_val_score(RF_clf,X, y, cv=5, scoring='accuracy')

print(aver_list)
print(aver_list2)
print(aver_list3)

import statistics
print(statistics.mean(aver_list))
print(statistics.stdev(aver_list))

print(statistics.mean(aver_list2))
print(statistics.stdev(aver_list2))

print(statistics.mean(aver_list3))
print(statistics.stdev(aver_list3))

f1_list = cross_val_score(clf, X, y, cv=5, scoring='f1')
f1_list_2 = cross_val_score(stupid_clf, X, y, cv=5, scoring='f1')
f1_list_3 = cross_val_score(RF_clf, X, y, cv=5, scoring='f1')


print(statistics.mean(f1_list))
print(statistics.stdev(f1_list))


print(statistics.mean(f1_list_2))
print(statistics.stdev(f1_list_2))


print(statistics.mean(f1_list_3))
print(statistics.stdev(f1_list_3))
# dot_data = tree.export_graphviz(clf, out_file="./random_forests_sklearn/vis/watermelon.dot") 
# graph = graphviz.Source(dot_data)
# graphviz.render(engine="dot",format="pdf",filepath="./random_forests_sklearn/vis/watermelon.dot") 

# random_forests_sklearn.decision_tree