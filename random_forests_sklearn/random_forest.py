from random_forests.utils import Dataset
from sklearn.ensemble import RandomForestClassifier

test_dataset = Dataset(_dataset_name = 'uci_blood', _dataset_file_path = './datasets/uci_blood.csv')
test_dataset.load_dataset(verbose=False)
X = test_dataset.samples[:,:-1].astype(int)
y = test_dataset.labels.astype(int)
clf = RandomForestClassifier(n_estimators = 100, criterion="entropy",oob_score=True)
clf = clf.fit(X,y)
print(clf.oob_score_)