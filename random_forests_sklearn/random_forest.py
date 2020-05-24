import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
# from random_forests.utils import Dataset
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve

# test_dataset = Dataset(_dataset_name = 'uci_blood', _dataset_file_path = './datasets/uci_blood.csv')
# test_dataset.load_dataset(verbose=False)
# X = test_dataset.samples[:,:-1].astype(int)
# y = test_dataset.labels.astype(int)
# clf = RandomForestClassifier(n_estimators = 100, criterion="entropy",oob_score=True)
# clf = clf.fit(X,y)
# print(clf.oob_score_)


def get_roc_curve():

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

    ax = plt.gca()

    # RandomForestClassifier
    RF_clf = RandomForestClassifier(n_estimators = 100, criterion="entropy",oob_score=True)
    RF_clf = RF_clf.fit(X_train,y_train)

    # DecisionTreeClassifier
    DT_clf = DecisionTreeClassifier(criterion='entropy')
    DT_clf = DT_clf.fit(X_train,y_train)

    # BaggingClassifier
    BAG_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy'),\
        n_estimators=100,oob_score=True)
    BAG_clf = BAG_clf.fit(X_train,y_train)

    # AdaBoostClassifier
    ADB_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy'),\
        n_estimators=100)
    ADB_clf = ADB_clf.fit(X_train,y_train)

    # GradientBoostingClassifier
    GB_clf = GradientBoostingClassifier(n_estimators=100)
    GB_clf = GB_clf.fit(X_train,y_train)

    
    rfc_disp = plot_roc_curve(RF_clf, X_test, y_test, ax=ax)
    dtc_disp = plot_roc_curve(DT_clf, X_test, y_test, ax=ax)
    bag_disp = plot_roc_curve(BAG_clf, X_test, y_test, ax=ax)
    adb_disp = plot_roc_curve(ADB_clf, X_test, y_test, ax=ax)
    gb_disp = plot_roc_curve(GB_clf, X_test, y_test, ax=ax)


    # rfc_disp.plot(ax=ax, alpha=0.8)
    # plt.show()

    plt.savefig("random_forests_sklearn/res/"+ dataset_name +"_2.pdf")
    plt.close()

    print("For "+dataset_name+ " ...")
    print("RandomForestClassifier ",RF_clf.oob_score_)
    print("BaggingClassifier ",BAG_clf.oob_score_)

    from sklearn.model_selection import cross_val_score
    import statistics

    print("RF")
    aver_list = cross_val_score(RF_clf, total_X, total_y, cv=10, scoring='accuracy')
    print(aver_list)
    print(statistics.mean(aver_list))
    print(statistics.stdev(aver_list))

    f1_list = cross_val_score(RF_clf, total_X, total_y, cv=10, scoring='f1')
    print(f1_list)
    print(statistics.mean(f1_list))
    print(statistics.stdev(f1_list))

    print("DT")
    aver_list = cross_val_score(DT_clf, total_X, total_y, cv=10, scoring='accuracy')
    print(aver_list)
    print(statistics.mean(aver_list))
    print(statistics.stdev(aver_list))

    f1_list = cross_val_score(DT_clf, total_X, total_y, cv=10, scoring='f1')
    print(f1_list)
    print(statistics.mean(f1_list))
    print(statistics.stdev(f1_list))

    print("BAG_clf")
    aver_list = cross_val_score(BAG_clf, total_X, total_y, cv=10, scoring='accuracy')
    print(aver_list)
    print(statistics.mean(aver_list))
    print(statistics.stdev(aver_list))

    f1_list = cross_val_score(BAG_clf, total_X, total_y, cv=10, scoring='f1')
    print(f1_list)
    print(statistics.mean(f1_list))
    print(statistics.stdev(f1_list))


    print("ADB_clf")
    aver_list = cross_val_score(ADB_clf, total_X, total_y, cv=10, scoring='accuracy')
    print(aver_list)
    print(statistics.mean(aver_list))
    print(statistics.stdev(aver_list))

    f1_list = cross_val_score(ADB_clf, total_X, total_y, cv=10, scoring='f1')
    print(f1_list)
    print(statistics.mean(f1_list))
    print(statistics.stdev(f1_list))
    

    print("GB_clf")
    aver_list = cross_val_score(GB_clf, total_X, total_y, cv=10, scoring='accuracy')
    print(aver_list)
    print(statistics.mean(aver_list))
    print(statistics.stdev(aver_list))

    f1_list = cross_val_score(GB_clf, total_X, total_y, cv=10, scoring='f1')
    print(f1_list)
    print(statistics.mean(f1_list))
    print(statistics.stdev(f1_list))

    # X_train, X_test, y_train, y_test = train_test_split(total_X, total_y, random_state=42)

    # ax = plt.gca()

    # # RandomForestClassifier
    # RF_clf = RandomForestClassifier(n_estimators = 100, criterion="entropy",oob_score=True)
    # RF_clf = RF_clf.fit(X_train,y_train)

    # # DecisionTreeClassifier
    # DT_clf = DecisionTreeClassifier(criterion='entropy')
    # DT_clf = DT_clf.fit(X_train,y_train)

    # # BaggingClassifier
    # BAG_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy'),\
    #     n_estimators=100,oob_score=True)
    # BAG_clf = BAG_clf.fit(X_train,y_train)

    # # AdaBoostClassifier
    # ADB_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy'),\
    #     n_estimators=100)
    # ADB_clf = ADB_clf.fit(X_train,y_train)

    # # GradientBoostingClassifier
    # GB_clf = GradientBoostingClassifier(n_estimators=100)
    # GB_clf = GB_clf.fit(X_train,y_train)


    # rfc_disp = plot_roc_curve(RF_clf, X_test, y_test, ax=ax)
    # dtc_disp = plot_roc_curve(DT_clf, X_test, y_test, ax=ax)
    # bag_disp = plot_roc_curve(BAG_clf, X_test, y_test, ax=ax)
    # adb_disp = plot_roc_curve(ADB_clf, X_test, y_test, ax=ax)
    # gb_disp = plot_roc_curve(GB_clf, X_test, y_test, ax=ax)

    # # rfc_disp.plot(ax=ax, alpha=0.8)
    # # plt.show()

    # plt.savefig("random_forests_sklearn/res/"+ "whole" +".pdf")
    # plt.close()

    # print("For "+ " whole "+ " ...")
    # print("RandomForestClassifier ",RF_clf.oob_score_)
    # print("BaggingClassifier ",BAG_clf.oob_score_)


def tune_hyper_parameter():

    dataset_path = "datasets/blood_donation/2016-04-08.npz"

    raw_data = np.load(dataset_path)
    X = raw_data['arr_0']
    y = raw_data['arr_1']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # # RandomForestClassifier
    # RF_clf = RandomForestClassifier(n_estimators = 100, criterion="entropy",oob_score=True)
    # RF_clf = RF_clf.fit(X_train,y_train)

    # print(RF_clf.score(X_test,y_test))

    # max_features hyperparameter
    ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, max_features='sqrt',
                                oob_score=True,
                               random_state=42)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=42)),
    ("RandomForestClassifier, max_features=None",
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
    plt.savefig("random_forests_sklearn/res/"+ "hyperpara_max_features_oob" +".pdf")
    plt.close()


    # Generate the "Mean Accuracy" vs. "n_estimators" plot.
    for label, mean_acc in mean_accuracy.items():
        xs, ys = zip(*mean_acc)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("Mean accuracy")
    plt.legend(loc="upper right")
    plt.savefig("random_forests_sklearn/res/"+ "hyperpara_max_features_mean_acc" +".pdf")
    plt.close()


    # criterion hyperparameter

    ensemble_clfs = [
    ("RandomForestClassifier, criterion='gini'",
        RandomForestClassifier(warm_start=True, criterion="gini",
                                oob_score=True,
                               random_state=42)),
    ("RandomForestClassifier, criterion='entropy'",
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
    plt.savefig("random_forests_sklearn/res/"+ "hyperpara_criterion_oob" +".pdf")
    plt.close()


    # Generate the "Mean Accuracy" vs. "n_estimators" plot.
    for label, mean_acc in mean_accuracy.items():
        xs, ys = zip(*mean_acc)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("Mean accuracy")
    plt.legend(loc="upper right")
    plt.savefig("random_forests_sklearn/res/"+ "hyperpara_criterion_mean_acc" +".pdf")
    plt.close()



def main():
    # tune_hyper_parameter()
    get_roc_curve()


if __name__ == "__main__":
    main()

# random_forests_sklearn.random_forest