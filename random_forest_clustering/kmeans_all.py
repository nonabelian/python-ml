import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def similarity(rf, X):
    nrows = X.shape[0]
    ncols = X.shape[0]
    sparse_sim = sparse.csr_matrix(np.zeros(shape=(nrows, ncols)))
    for clf in rf.estimators_:
        indices = clf.apply(X)

        df_indices = pd.DataFrame({'indices': indices}).reset_index()
        df_sims = df_indices.merge(df_indices, on='indices')

        rows = df_sims['index_x']
        cols = df_sims['index_y']
        data= np.ones(len(df_sims))
        sim = sparse.csr_matrix((data, (rows, cols)), shape=(nrows, ncols))

        sparse_sim += sim

    sparse_sim /= float(len(rf.estimators_))

    return sparse_sim


def prepare_data(num_add=3):
    data = load_iris()

    df_X_iris = pd.DataFrame(data['data'], columns=data['feature_names'])
    df_X_iris['target'] = 1
    df_X_iris['true_class_label'] = data['target']

    dfs_add = [df_X_iris.copy()]
    for i in range(num_add):
        df_X_iris_add = pd.DataFrame(data['data'],
                                     columns=data['feature_names'])
        df_X_iris_add += 0.1 * (np.random.rand(df_X_iris_add.shape[0]
                                                * df_X_iris_add.shape[1])
                                 .reshape(df_X_iris_add.shape) - 0.5)
        df_X_iris_add['target'] = 1
        df_X_iris_add['true_class_label'] = data['target']
        dfs_add.append(df_X_iris_add.copy())

    df_X_iris = pd.concat(dfs_add, axis=0)

    df_X_notiris = df_X_iris[data['feature_names']]\
            .apply(lambda x: np.random.permutation(x))
    df_X_notiris['target'] = 0
    df_X_notiris['true_class_label'] = -1
    df_X = pd.concat([df_X_iris, df_X_notiris], axis=0)

    return df_X.reset_index(drop=True)


if __name__ == '__main__':

    data = load_iris()

    df_X_iris = pd.DataFrame(data['data'], columns=data['feature_names'])
    df_X_iris['target'] = data['target']

    X = df_X_iris[data['feature_names']].values
    y = df_X_iris['target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    print('=' * 30)
    print(rf.score(X_test, y_test))
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('=' * 30)

    # Separate out the original dataset
    df_X_labeled = df_X_iris.copy()
    df_X_labeled['target'] = data['target']

    ## Descriminator

    # Augmented dataset:
    df_X = prepare_data(num_add=4)
    X = df_X.drop(columns=['target', 'true_class_label'])
    y = df_X['target']
    X_train, X_test, y_train, y_test, train_idx, test_idx = \
            train_test_split(X, y, df_X.index, test_size=0.2,
                             stratify=y, random_state=42)

    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(X_train, y_train)
    print('Descriminator Score: ', rf.score(X_test, y_test))

    df_X_labeled = pd.DataFrame(data['data'], columns=data['feature_names'])
    df_X_labeled['target'] = data['target']

    df_X_labeled_all = df_X[df_X['target'] == 1].copy()
    df_X_labeled_all['target'] = df_X_labeled_all['true_class_label']
    df_X_labeled_all.drop(columns=['true_class_label'], inplace=True)
    # Work off the whole dataset for similarity:

    # Fit on whole set and predict missing classes:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=3)
    km.fit(df_X_labeled_all[data['feature_names']],
           df_X_labeled_all['target'])
    y_pred = km.predict(df_X_labeled_all[data['feature_names']].values)
    df_X_labeled_all['cluster_label'] = y_pred

    target_to_label = {}
    for target in df_X_labeled_all['target'].unique().tolist():
        label = df_X_labeled_all[df_X_labeled_all['target'] == target]['cluster_label']\
                    .value_counts()\
                    .index[0]
        target_to_label[target] = label

    labels_to_target = {v: k for k, v in target_to_label.items()}

    df_X_labeled_all['y_pred'] = df_X_labeled_all['cluster_label']\
            .apply(lambda x: labels_to_target[x])

    print(classification_report(df_X_labeled_all['target'].values,
                                df_X_labeled_all['y_pred'].values))
    print(confusion_matrix(df_X_labeled_all['target'].values,
                           df_X_labeled_all['y_pred'].values))
    print("Accuracy: ", accuracy_score(df_X_labeled_all['target'].values,
                                       df_X_labeled_all['y_pred'].values))
    df_X_labeled_all['cluster_label'].value_counts()
