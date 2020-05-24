import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
# from random_forests.utils import Dataset
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import cross_val_score
# test_dataset = Dataset(_dataset_name = 'uci_blood', _dataset_file_path = './datasets/uci_blood.csv')
# test_dataset.load_dataset(verbose=False)
# X = test_dataset.samples[:,:-1].astype(int)
# y = test_dataset.labels.astype(int)
# clf = RandomForestClassifier(n_estimators = 100, criterion="entropy",oob_score=True)
# clf = clf.fit(X,y)
# print(clf.oob_score_)


def hyper_out():

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

            if dataset_name != '2016-04-08' and dataset_name != '2016-04-18' and dataset_name != '2016-04-28' and dataset_name != '2016-05-20' and dataset_name != '2016-05-21' and dataset_name != '2016-06-04'\
                and dataset_name != '2016-06-12':
                continue

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

    # RandomForestClassifier
    RF_clf = RandomForestClassifier(n_estimators = 100, criterion="entropy",oob_score=True)
    RF_clf.fit(total_X,total_y)
    print("oob:",RF_clf.oob_score_)
    aver_list = cross_val_score(RF_clf, total_X, total_y, cv=10, scoring='accuracy')

    import statistics
    print(aver_list)
    print(statistics.mean(aver_list))
    print(statistics.stdev(aver_list))


    ensemble_clfs = [
    ("CART",
        RandomForestClassifier(n_estimators=100,warm_start=True, criterion="gini",
                                oob_score=True,
                               random_state=42)),
    ("C4.5",
        RandomForestClassifier(n_estimators=100,warm_start=True, criterion="entropy",
                               oob_score=True,
                               random_state=42))
    ]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
    mean_accuracy = OrderedDict((label, []) for label, _ in ensemble_clfs)


    # Range of `n_estimators` values to explore.
    min_per = 5
    max_per = 100

    X_train, X_test, y_train, y_test = train_test_split(total_X, total_y, random_state=42,test_size=0.2)
    num_samples = X_train.shape[0]
    
    for label, clf in ensemble_clfs:
        for i in range(min_per, max_per + 1, 5):
            # clf.set_params(n_estimators=100)
            
            X = total_X[0:num_samples*i // 100,:]
            y = total_y[0:num_samples*i // 100]

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,test_size=0.1)

            clf.fit(X_train, y_train)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            print(oob_error)
            error_rate[label].append((i, oob_error))

            # Record mean accuracy for each `n_estimators=i` setting.
            mean_acc = clf.score(X_test,y_test)
            mean_accuracy[label].append((i,mean_acc))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_per, max_per)
    plt.xlabel("Training data usage percentage")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.savefig("random_forests_sklearn/res/"+ "hyperpara_out_oob" +".pdf")
    plt.close()


    # Generate the "Mean Accuracy" vs. "n_estimators" plot.
    for label, mean_acc in mean_accuracy.items():
        xs, ys = zip(*mean_acc)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_per, max_per)
    plt.xlabel("Training data usage percentage")
    plt.ylabel("Mean accuracy")
    plt.legend(loc="upper right")
    plt.savefig("random_forests_sklearn/res/"+ "hyperpara_out_mean_acc" +".pdf")
    plt.close()

def inner_hyper():
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

            if dataset_name != '2016-04-08' and dataset_name != '2016-04-18' and dataset_name != '2016-04-28' and dataset_name != '2016-05-20' and dataset_name != '2016-05-21' and dataset_name != '2016-06-04'\
                and dataset_name != '2016-06-12':
                continue

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

    # RandomForestClassifier
    RF_clf = RandomForestClassifier(n_estimators = 100, criterion="entropy",oob_score=True)
    RF_clf.fit(total_X,total_y)
    print("oob:",RF_clf.oob_score_)
    aver_list = cross_val_score(RF_clf, total_X, total_y, cv=10, scoring='accuracy')

    import statistics
    print(aver_list)
    print(statistics.mean(aver_list))
    print(statistics.stdev(aver_list))


    ensemble_clfs = [
    ("CART",
        RandomForestClassifier(n_estimators=100,warm_start=True, criterion="gini",
                                oob_score=True,
                               random_state=42)),
    ("C4.5",
        RandomForestClassifier(n_estimators=100,warm_start=True, criterion="entropy",
                               oob_score=True,
                               random_state=42))
    ]

    # max_features hyperparameter
    ensemble_clfs = [
    ("sqrt",
        RandomForestClassifier(warm_start=True, max_features='sqrt',
                                oob_score=True,
                               random_state=42)),
    ("log2",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=42)),
    ("all",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=42))
    ]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
    mean_accuracy = OrderedDict((label, []) for label, _ in ensemble_clfs)


    # Range of `n_estimators` values to explore.
    min_estimators = 10
    max_estimators = 200

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X_train, y_train)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

            # Record mean accuracy for each `n_estimators=i` setting.
            mean_acc = clf.score(X_test,y_test)
            mean_accuracy[label].append((i,mean_acc))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.savefig("random_forests_sklearn/res/"+ "hyperpara_max_features_oob_2" +".pdf")
    plt.close()


    # Generate the "Mean Accuracy" vs. "n_estimators" plot.
    for label, mean_acc in mean_accuracy.items():
        xs, ys = zip(*mean_acc)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("Mean accuracy")
    plt.legend(loc="upper right")
    plt.savefig("random_forests_sklearn/res/"+ "hyperpara_max_features_mean_acc_2" +".pdf")
    plt.close()


    # criterion hyperparameter

    ensemble_clfs = [
    ("CART",
        RandomForestClassifier(warm_start=True, criterion="gini",
                                oob_score=True,
                               random_state=42)),
    ("C4.5",
        RandomForestClassifier(warm_start=True, criterion="entropy",
                               oob_score=True,
                               random_state=42))
    ]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
    mean_accuracy = OrderedDict((label, []) for label, _ in ensemble_clfs)


    # Range of `n_estimators` values to explore.
    min_estimators = 10
    max_estimators = 200

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X_train, y_train)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

            # Record mean accuracy for each `n_estimators=i` setting.
            mean_acc = clf.score(X_test,y_test)
            mean_accuracy[label].append((i,mean_acc))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.savefig("random_forests_sklearn/res/"+ "hyperpara_criterion_oob_2" +".pdf")
    plt.close()


    # Generate the "Mean Accuracy" vs. "n_estimators" plot.
    for label, mean_acc in mean_accuracy.items():
        xs, ys = zip(*mean_acc)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("Mean accuracy")
    plt.legend(loc="upper right")
    plt.savefig("random_forests_sklearn/res/"+ "hyperpara_criterion_mean_acc_2" +".pdf")
    plt.close()


def performance():
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

            if dataset_name != '2016-04-08' and dataset_name != '2016-04-18' and dataset_name != '2016-04-28' and dataset_name != '2016-05-20' and dataset_name != '2016-05-21' and dataset_name != '2016-06-04'\
                and dataset_name != '2016-06-12':
                continue

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

    # RandomForestClassifier
    RF_clf = RandomForestClassifier(n_estimators = 100, criterion="entropy",oob_score=True)
    RF_clf.fit(X_train,y_train)
    print("RF")
    print("oob:",RF_clf.oob_score_)
    aver_list = cross_val_score(RF_clf, total_X, total_y, cv=10, scoring='accuracy')

    import statistics
    print(aver_list)
    print(statistics.mean(aver_list))
    print(statistics.stdev(aver_list))

    f1_list = cross_val_score(RF_clf, total_X, total_y, cv=10, scoring='f1')
    print(f1_list)
    print(statistics.mean(f1_list))
    print(statistics.stdev(f1_list))

    from sklearn import svm
    print("SVM")
    svm_clf = svm.SVC()
    svm_clf.fit(X_train,y_train)
    aver_list_2 = cross_val_score(svm_clf, total_X, total_y, cv=10, scoring='accuracy')

    import statistics
    print(aver_list_2)
    print(statistics.mean(aver_list_2))
    print(statistics.stdev(aver_list_2))

    f1_list = cross_val_score(svm_clf, total_X, total_y, cv=10, scoring='f1')
    print(f1_list)
    print(statistics.mean(f1_list))
    print(statistics.stdev(f1_list))


    from sklearn.neighbors import KNeighborsClassifier
    print("kNN")
    knn_clf = KNeighborsClassifier(n_neighbors=3)
    knn_clf.fit(X_train,y_train)
    aver_list_3 = cross_val_score(knn_clf, total_X, total_y, cv=10, scoring='accuracy')

    import statistics
    print(aver_list_3)
    print(statistics.mean(aver_list_3))
    print(statistics.stdev(aver_list_3))

    f1_list = cross_val_score(knn_clf, total_X, total_y, cv=10, scoring='f1')
    print(f1_list)
    print(statistics.mean(f1_list))
    print(statistics.stdev(f1_list))

    from sklearn.neural_network import MLPClassifier
    print("NN")
    nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    nn_clf.fit(X_train,y_train)
    aver_list_4 = cross_val_score(nn_clf, total_X, total_y, cv=10, scoring='accuracy')

    import statistics
    print(aver_list_4)
    print(statistics.mean(aver_list_4))
    print(statistics.stdev(aver_list_4))

    f1_list = cross_val_score(nn_clf, total_X, total_y, cv=10, scoring='f1')
    print(f1_list)
    print(statistics.mean(f1_list))
    print(statistics.stdev(f1_list))


    from sklearn.linear_model import SGDClassifier
    print('SGDClassifier')
    SGD_clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=50)
    SGD_clf.fit(X_train,y_train)
    aver_list_5 = cross_val_score(SGD_clf, total_X, total_y, cv=10, scoring='accuracy')

    import statistics
    print(aver_list_5)
    print(statistics.mean(aver_list_5))
    print(statistics.stdev(aver_list_5))

    f1_list = cross_val_score(SGD_clf, total_X, total_y, cv=10, scoring='f1')
    print(f1_list)
    print(statistics.mean(f1_list))
    print(statistics.stdev(f1_list))


    ax = plt.gca()
    RF_clf_disp = plot_roc_curve(RF_clf, X_test, y_test, ax=ax)
    svm_clf_disp = plot_roc_curve(svm_clf, X_test, y_test, ax=ax)
    knn_clf_disp = plot_roc_curve(knn_clf, X_test, y_test, ax=ax)
    nn_clf_disp = plot_roc_curve(nn_clf, X_test, y_test, ax=ax)
    SGD_clf_disp = plot_roc_curve(SGD_clf, X_test, y_test, ax=ax)


    # rfc_disp.plot(ax=ax, alpha=0.8)
    # plt.show()

    plt.savefig("random_forests_sklearn/res/"+ "inter_whole_roc" +".pdf")
    plt.close()

performance()
# inner_hyper()
# random_forests_sklearn.out_hyper