import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ecg_classify.constants import FEATURE_NUM
from ecg_classify.data_io import read_data
from sklearn.utils import shuffle


def __load_data(force=False):
    start_time = time.time()
    df_train, df_test = read_data(force)
    end_time = time.time()
    print("consume time: %.2f s" % (end_time - start_time))
    return df_train, df_test


def transform_label_to_num(y_label):
    y = np.zeros(len(y_label), dtype=int)
    for i in range(len(y_label)):
        if y_label[i] == 'N':
            y[i] = 0
        elif y_label[i] == 'L':
            y[i] = 1
        elif y_label[i] == 'R':
            y[i] = 2
        elif y_label[i] == 'A':
            y[i] = 3
        elif y_label[i] == 'V':
            y[i] = 4
        else:
            y[i] = 5
    return y


def __prepare_data(df_train, df_test, inter=True):
    X_train = df_train.drop(str(FEATURE_NUM), axis=1).values
    X_test = df_test.drop(str(FEATURE_NUM), axis=1).values
    y_train_label = df_train[str(FEATURE_NUM)].values
    y_test_label = df_test[str(FEATURE_NUM)].values
    y_train = transform_label_to_num(y_train_label)
    y_test = transform_label_to_num(y_test_label)

    # y_train[y_train == 2] = 1
    # y_test[y_test == 2] = 1
    if inter:
        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)
    else:
        X = np.vstack((X_train, X_test))
        y = np.concatenate((y_train, y_test))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test


def inter_patient():
    df_train, df_test = __load_data()
    X_train, y_train, X_test, y_test = __prepare_data(df_train, df_test, True)

    clf = RandomForestClassifier(n_estimators=200, max_depth=7)
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    print("training time span: %.2f s" % (end_time - start_time))
    y_pred = clf.predict(X_test)

    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    y_pred_all = clf.predict(X)

    print(accuracy_score(y_true=y_test, y_pred=y_pred))
    print(confusion_matrix(y_true=y_test, y_pred=y_pred))

    print(accuracy_score(y_true=y, y_pred=y_pred_all))
    print(confusion_matrix(y_true=y, y_pred=y_pred_all))


# def intra_patient():
#     df_train, df_test = __load_data()
#     X_train, y_train, X_test, y_test = __prepare_data(
#         df_train, df_test, MultiLabel(), inter=False)
#
#     clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=0)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print(accuracy_score(y_true=y_test, y_pred=y_pred))
#     print(confusion_matrix(y_true=y_test, y_pred=y_pred))


# def compute_dataset_shift():
#     df_train, df_test = __load_data()
#     X_train, X_test, y_train, y_test = __prepare_data(
#         df_train, df_test, SingleLabel(), inter=False)
#     clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print(accuracy_score(y_true=y_test, y_pred=y_pred))
#     print(confusion_matrix(y_true=y_test, y_pred=y_pred))
#     plot_feature_importance(df_train, clf)


def plot_feature_importance(df_train, clf):
    features = df_train.columns.values
    imp = clf.feature_importances_
    indices = np.argsort(imp)[::-1][:8]

    # plot
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(indices)), imp[indices], color='b', align='center')
    plt.xticks(range(len(indices)), features[indices], rotation='vertical')
    plt.xlim([-1,len(indices)])
    plt.show()


def normalize_test(X_train, X_test):
    # normalize test
    mean_n = X_train[0: 4000].sum(axis=0) / 4000
    mean_l = X_train[4000: 8000].sum(axis=0) / 4000
    mean_r = X_train[8000: 12000].sum(axis=0) / 4000
    mean_a = X_train[12000: 16000].sum(axis=0) / 4000
    mean_v = X_train[16000: 20000].sum(axis=0) / 4000

    mean_n_test = X_test[0: 1000].sum(axis=0) / 1000
    mean_l_test = X_test[1000: 2000].sum(axis=0) / 1000
    mean_r_test = X_test[2000: 3000].sum(axis=0) / 1000
    mean_a_test = X_test[3000: 4000].sum(axis=0) / 1000
    mean_v_test = X_test[4000: 5000].sum(axis=0) / 1000

    X_test[0: 1000] = X_test[0: 1000] * mean_n / mean_n_test
    X_test[1000: 2000] = X_test[1000: 2000] * mean_l / mean_l_test
    X_test[2000: 3000] = X_test[2000: 3000] * mean_r / mean_r_test
    X_test[3000: 4000] = X_test[3000: 4000] * mean_a / mean_a_test
    X_test[4000: 5000] = X_test[4000: 5000] * mean_v / mean_v_test
