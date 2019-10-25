from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from ecg_classify.feature import get_features, get_labels

X = get_features(True)
X_test = get_features(False)
y = get_labels(True)
y_test = get_labels(False)

# scale training sample
X_scaler = StandardScaler().fit_transform(X)

# select feature
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X_scaler)

decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
decision_tree = decision_tree.fit(X, y)

