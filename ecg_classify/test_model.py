import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import pydot

from ecg_classify.constants import FEATURE_NUM
from ecg_classify.gen_data import gen_label, read_data
from ecg_classify.utils import shuffle_data
from sklearn.ensemble import VotingClassifier
from sklearn.utils import shuffle
from scipy import stats

import os
os.chdir('./ecg_classify')

df_train, df_test = read_data(True)
X_train1 = df_train[['0', '1', '2', '3', '4', '5', '6', '7']].values
X_test1 = df_test[['0', '1', '2', '3', '4', '5', '6', '7']].values
X_train2 = df_train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']].values
X_test2 = df_test[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']].values
X_train = df_train.drop(str(FEATURE_NUM - 1), axis=1).values
X_test = df_test.drop(str(FEATURE_NUM - 1), axis=1).values

y_train = gen_label(True)
y_test = gen_label(False)

X_train, y_train, X_train1, X_train2 = shuffle(X_train, y_train, X_train1, X_train2)
clf0 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
clf1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
clf2 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
clf0.fit(X_train, y_train)
clf1.fit(X_train1, y_train)
clf2.fit(X_train2, y_train)

# eclf = VotingClassifier(estimators=[('rf', clf0), ('rf', clf1), ('rf', clf2)], voting='hard')

# y_pred = eclf.predict(X_test)
y_pred0 = clf0.predict(X_test)
y_pred1 = clf1.predict(X_test1)
y_pred2 = clf2.predict(X_test2)

print(accuracy_score(y_true=y_test, y_pred=y_pred0))
print(accuracy_score(y_true=y_test, y_pred=y_pred1))
print(accuracy_score(y_true=y_test, y_pred=y_pred2))

print(confusion_matrix(y_true=y_test, y_pred=y_pred0))
print(confusion_matrix(y_true=y_test, y_pred=y_pred1))
print(confusion_matrix(y_true=y_test, y_pred=y_pred2))

y_pred = clf0.predict(X_train)
print(confusion_matrix(y_true=y_train, y_pred=y_train))

# X = np.concatenate((X_train, X_test))
# y = np.concatenate((y_train, y_test))
# y_ = clf.predict(X)
# print(accuracy_score(y_true=y, y_pred=y_))
t_kur_1 = df_train[0: 4000]['0']
t_kur_2 = df_train[4000: 8000]['0']
t_kur_3 = df_train[8000: 12000]['0']
t_kur_4 = df_train[12000: 16000]['0']
t_kur_5 = df_train[16000: 20000]['0']

t_kur_1 = df_test[0: 1000]['0']
t_kur_2 = df_test[1000: 2000]['0']
t_kur_3 = df_test[2000: 3000]['0']
t_kur_4 = df_test[3000: 4000]['0']
t_kur_5 = df_test[4000: 5000]['0']


def vote_res(y1, y2, y3):
    n = len(y1)
    y = np.vstack([y1, y2, y3])
    res = np.zeros(n)
    for i in range(n):
        res[i] = stats.mode(y[:, i])[0][0]
    return res


# compute dataset shift
def compute_dataset_shift():
    df_train, df_test = read_data()
    X_train = df_train.drop('8', axis=1).values
    X_test = df_test.drop('8', axis=1).values
    y_train = np.full(20000, 0)
    y_test = np.full(5000, 1)
    X = np.vstack((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    clf.fit(X_tr, y_tr)
    y_pr = clf.predict(X_te)
    print(accuracy_score(y_true=y_te, y_pred=y_pr))
    print(confusion_matrix(y_true=y_te, y_pred=y_pr))
    plot_feature_importance(df_train, clf)


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
