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
    rf.fit(X, y)
    sparse_sim = similarity(rf, df_X_labeled_all.drop(columns=['target']).values)
    sparse_dist = sparse_sim.copy()
    sparse_dist.data = 1.0 / sparse_dist.data
    sparse_dist.setdiag(0)
    sparse_dist.eliminate_zeros()

    # Cluster based on the RF similarities:
    from sklearn.cluster import DBSCAN
    dbs = DBSCAN(eps=2.3, metric='precomputed')
    dbs.fit(sparse_dist)
    print(dbs.labels_)

    df_X_labeled_all['cluster_label'] = dbs.labels_
    df_X_labeled_all.head()

    # Create the mapping for comparison
    target_to_label = {}
    for target in df_X_labeled_all['target'].unique().tolist():
        label = df_X_labeled_all[df_X_labeled_all['target'] == target]['cluster_label']\
                    .value_counts()\
                    .index[0]
        if label == -1:
            label = df_X_labeled_all[df_X_labeled_all['target'] == target]\
                    ['cluster_label']\
                    .value_counts()\
                    .index[1]
        target_to_label[target] = label

    for target in df_X_labeled_all['target'].unique().tolist():
        print('Target: ', target)
        df_target = df_X_labeled_all[df_X_labeled_all['target'] == target]
        label_count = df_target[df_target['cluster_label'] ==
                                target_to_label[target]].shape[0]
        overlap = label_count / df_target.shape[0]
        print(overlap)
    print(target_to_label)

    # Build classifier for the clusters (cluster prediction):
    df_X_nn = df_X_labeled_all[df_X_labeled_all['cluster_label']
                           .isin(target_to_label.values())]
    df_X_nn_predict = df_X_labeled_all[~df_X_labeled_all['cluster_label']
                                   .isin(target_to_label.values())]
    predict_indices = df_X_nn_predict.index

    X_nn = df_X_nn[data['feature_names']].values
    y_nn = df_X_nn['cluster_label']

    # Check performance:
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = \
            train_test_split(X_nn, y_nn, test_size=0.3)
    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(X_train_nn, y_train_nn)
    print('=' * 30)
    print("RF test classification score: ", rf.score(X_test_nn, y_test_nn))
    print(classification_report(y_test_nn, rf.predict(X_test_nn)))
    print('=' * 30)

    # Fit on whole set and predict missing classes:
    rf.fit(X_nn, y_nn)
    y_pred_nn = rf.predict(df_X_nn_predict[data['feature_names']].values)
    df_X_labeled_all.loc[predict_indices, 'cluster_label'] = y_pred_nn

    for target in df_X_labeled_all['target'].unique().tolist():
        print('Target: ', target)
        df_target = df_X_labeled_all[df_X_labeled_all['target'] == target]
        label_count = df_target[df_target['cluster_label'] ==
                                target_to_label[target]].shape[0]
        overlap = label_count / df_target.shape[0]
        print(overlap)

    labels_to_target = {v: k for k, v in target_to_label.items()}
    for label in df_X_labeled_all['cluster_label'].unique().tolist():
        if label not in target_to_label.values():
            labels_to_target[label] = -1

    ## Plotting:
    colors = ['r' if l == 0 else 'b' if l == 1 else 'g'
              for l in df_X_labeled_all['target']]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    X_plot = df_X_labeled_all[data['feature_names']].values
    ax.scatter(X_plot[:, 0], X_plot[:, 1], color=colors)

    colors_labeled = ['r' if labels_to_target[l] == 0 else
                      'b' if labels_to_target[l] == 1 else
                      'g' if labels_to_target[l] == 2 else
                      'black'
                      for l in df_X_labeled_all['cluster_label']]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(X_plot[:, 0], X_plot[:, 1], color=colors_labeled)

    plt.show()

    ## Reports:
    df_X_labeled_all['y_pred'] = df_X_labeled_all['cluster_label']\
            .apply(lambda x: labels_to_target[x])

    print(classification_report(df_X_labeled_all['target'].values,
                                df_X_labeled_all['y_pred'].values))
    print(confusion_matrix(df_X_labeled_all['target'].values,
                           df_X_labeled_all['y_pred'].values))
    print("Accuracy: ", accuracy_score(df_X_labeled_all['target'].values,
                                       df_X_labeled_all['y_pred'].values))
    df_X_labeled_all['cluster_label'].value_counts()
