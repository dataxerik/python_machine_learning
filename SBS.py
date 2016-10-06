from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd

class SBS():
    def __init__(self, estimator, k_features, scoring = accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.test_size,
                                                            random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)

        return score

if __name__ == '__main__':
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                          header=None)
    df_wine.columns = [
        'Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
        'Total phenois', 'Flavanoids', 'Nonfalvanoid phenols', 'Proanthocyanins', 'Color intensity',
        'Hue', 'OD280/OD315 of diluted wines', 'Proline'
    ]

    print('Class labels', np.unique(df_wine['Class label']))

    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.fit_transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=2)

    # selectign features
    sbs = SBS(knn, k_features=1)
    sbs.fit(X_train_std, y_train)

    k_feat = [len(k) for k in sbs.subsets_]

    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.1])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.show()

    k5 = list(sbs.subsets_[8])
    print(df_wine.columns[1:][k5])

    knn.fit(X_train_std, y_train)
    print('Training accuracy: ', knn.score(X_train_std, y_train))
    print('Testing accuracy: ', knn.score(X_test_std, y_test))

    knn.fit(X_train_std[:, k5], y_train)
    print('Training accuracy: ', knn.score(X_train_std[:, k5], y_train))
    print('Testing accuracy: ', knn.score(X_test_std[:, k5], y_test))

    feat_labels = df_wine.columns[1:]

    forest = RandomForestClassifier(n_estimators=10000,
                                    random_state=0,
                                    n_jobs=-1)

    forest.fit(X_train, y_train)
    importances = forest.feature_importances_

    indicies = np.argsort(importances)[::-1]

    for f in range(X_train.shape[1]):
        print("%2d) % -*s %f" % (f + 1, 30,
                                 feat_labels[indicies[f]],
                                 importances[indicies[f]]))
    plt.title("Feature Importance")
    plt.bar(range(X_train.shape[1]),
            importances[indicies],
            color='lightblue',
            align='center')
    plt.xticks(range(X_train.shape[1]),
               feat_labels[indicies], rotation=90)

    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    plt.show()

    X_selected = forest.transform(X_train, threshold=0.15)
    print(X_selected.shape)
    for f in range(X_selected.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

